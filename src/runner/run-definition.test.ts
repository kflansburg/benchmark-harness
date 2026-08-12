import { describe, expect, it } from "bun:test";

import { fromIterable } from "effect/Chunk";
import {
  dieMessage,
  fail as effectFail,
  gen,
  map,
  sleep,
  succeed as effectSucceed,
  sync,
} from "effect/Effect";
import type { Layer } from "effect/Layer";
import {
  effect as layerEffect,
  fail as layerFail,
  mergeAll as layerMergeAll,
  provide as layerProvide,
  succeed as layerSucceed,
} from "effect/Layer";
import { fail as streamFail, fromChunk } from "effect/Stream";

import type {
  BenchmarkConfig,
  GpqaBenchmarkConfig,
} from "../benchmarks/benchmark-config";
import { defineChatBenchmark } from "../benchmarks/define-chat-benchmark";
import type { Benchmark } from "../benchmarks/types";
import type { Sample } from "../harness/core";
import {
  DatasetError,
  MessageRole,
  ModelError,
  ScoreValue,
} from "../harness/core";
import { Dataset } from "../harness/dataset";
import { Model } from "../harness/model";
import type {
  CheckpointStoreService,
  ProgressReporterService,
} from "../harness/progress";
import { Scorer } from "../harness/scorer";
import { generate, Solver } from "../harness/solver";
import { Either } from "../internal/either";
import { ResponsesModel } from "../providers/responses-model";
import { runBenchmarkDefinition } from "./run-definition";

function datasetLayer(samples: readonly Sample[]): Layer<Dataset> {
  return layerSucceed(
    Dataset,
    Dataset.of({
      stream: (opts) => {
        const start = opts?.start ?? 0;
        const end = opts?.end ?? samples.length;
        return fromChunk(fromIterable(samples.slice(start, end)));
      },
      size: effectSucceed(samples.length),
    })
  );
}

const samples = Array.from({ length: 4 }, (_, index) => ({
  id: `sample-${index}`,
  input: `Question ${index}`,
  target: { text: "correct" },
}));

const chatBenchmark = defineChatBenchmark<GpqaBenchmarkConfig>({
  id: "gpqa_diamond",
  temperature: 0,
  defaultEpochs: 1,
  isConfig: (config): config is GpqaBenchmarkConfig =>
    config.benchmarkId === "gpqa_diamond",
  makeDatasetLayer: () => datasetLayer(samples.slice(0, 1)),
  scorer: (_state, target) =>
    effectSucceed({
      value: ScoreValue.Correct,
      answer: target.text,
      explanation: "correct",
    }),
  makeSolver: (model) => generate(model, { temperature: 0 }),
});

interface ResponsesTestConfig extends BenchmarkConfig {
  readonly benchmarkId: "responses-test";
  readonly model: string;
  readonly transport: "responses";
}

describe("Worker runtime", () => {
  it("runs without a process global", async () => {
    const processDescriptor = Object.getOwnPropertyDescriptor(
      globalThis,
      "process"
    );
    const modelLayer = layerSucceed(
      Model,
      Model.of({
        generate: () =>
          effectSucceed({
            completion: "correct",
            message: {
              role: MessageRole.Assistant,
              content: "correct",
            },
          }),
      })
    );
    Reflect.deleteProperty(globalThis, "process");
    try {
      const output = await runBenchmarkDefinition({
        benchmark: chatBenchmark,
        benchmarkConfig: {
          benchmarkId: "gpqa_diamond",
          model: "injected-model",
        },
        sessionId: "worker-runtime",
        modelLayer,
        datasetLayer: datasetLayer(samples.slice(0, 1)),
        resultStore: {
          write: () => dieMessage("persistence unavailable"),
        },
        epochs: 1,
        maxConcurrency: 1,
      });
      expect(Either.isRight(output)).toBe(true);
    } finally {
      if (processDescriptor !== undefined) {
        Object.defineProperty(globalThis, "process", processDescriptor);
      }
    }
  });
});

describe("runBenchmarkDefinition", () => {
  it("runs a supplied benchmark and runtime services without the registry", async () => {
    let active = 0;
    let maxActive = 0;
    const modelLayer = layerSucceed(
      Model,
      Model.of({
        generate: () =>
          gen(function* () {
            active += 1;
            maxActive = Math.max(maxActive, active);
            yield* sleep("10 millis");
            active -= 1;
            return {
              completion: "correct",
              message: {
                role: MessageRole.Assistant,
                content: "correct",
              },
            };
          }),
      })
    );
    const progressEvents: string[] = [];
    const progressReporter: ProgressReporterService = {
      onSampleStart: (event) =>
        sync(() => {
          progressEvents.push(`start:${event.sampleIndex}`);
        }),
      onSampleEnd: (event) =>
        sync(() => {
          progressEvents.push(`end:${event.sampleId}`);
        }),
      onSampleComplete: (count) =>
        sync(() => {
          progressEvents.push(`complete:${count}`);
        }),
      onAgentStep: () => effectSucceed(undefined),
    };
    const checkpointStore: CheckpointStoreService = {
      read: async () => null,
      write: async () => {},
      remove: async () => {},
    };
    const output = await runBenchmarkDefinition({
      benchmark: chatBenchmark,
      benchmarkConfig: {
        benchmarkId: "gpqa_diamond",
        model: "injected-model",
      },
      sessionId: "session",
      modelLayer,
      datasetLayer: datasetLayer(samples),
      progressReporter,
      checkpointStore,
      range: { start: 1, end: 3 },
      epochs: 2,
      maxConcurrency: 2,
    });

    expect(Either.isRight(output)).toBe(true);
    if (Either.isLeft(output)) {
      throw new Error(output.left);
    }
    expect(
      output.right.result.sampleScores.map((score) => score.sampleId)
    ).toEqual(["sample-1", "sample-1", "sample-2", "sample-2"]);
    expect(output.right.resultsPath).toBeNull();
    expect(maxActive).toBe(2);
    expect(
      progressEvents.filter((event) => event.startsWith("start:"))
    ).toHaveLength(4);
    expect(progressEvents).toContain("complete:4");
  });

  it("returns a credential error instead of constructing a malformed request", async () => {
    const output = await runBenchmarkDefinition({
      benchmark: chatBenchmark,
      benchmarkConfig: {
        benchmarkId: "gpqa_diamond",
        model: "default-model",
      },
      sessionId: "missing-credential",
      apiKey: "",
      datasetLayer: datasetLayer(samples.slice(0, 1)),
      epochs: 1,
      maxConcurrency: 1,
    });

    expect(Either.isLeft(output)).toBe(true);
    if (Either.isRight(output)) {
      throw new Error("expected the run to fail");
    }
    expect(output.left).toEqual({
      category: "internal",
      retryable: false,
      systemic: true,
    });
  });

  it("returns safe structured model and dataset failures without messages", async () => {
    const secret = "SECRET_PROVIDER_OR_DATASET_DETAIL";
    const modelOutput = await runBenchmarkDefinition({
      benchmark: chatBenchmark,
      benchmarkConfig: { benchmarkId: "gpqa_diamond", model: "injected-model" },
      sessionId: "model-error",
      modelLayer: layerSucceed(
        Model,
        Model.of({
          generate: () =>
            effectFail(
              new ModelError({ message: secret, status: 503, systemic: true })
            ),
        })
      ),
      datasetLayer: datasetLayer(samples.slice(0, 1)),
      epochs: 1,
      maxConcurrency: 1,
    });
    expect(Either.isLeft(modelOutput)).toBe(true);
    if (Either.isRight(modelOutput)) throw new Error("expected model failure");
    expect(modelOutput.left).toEqual({
      category: "model",
      status: 503,
      retryable: true,
      systemic: true,
    });
    expect(JSON.stringify(modelOutput.left)).not.toContain(secret);

    const datasetError = Object.assign(new DatasetError({ message: secret }), {
      code: "transient",
      retryable: true,
    });
    const datasetOutput = await runBenchmarkDefinition({
      benchmark: chatBenchmark,
      benchmarkConfig: { benchmarkId: "gpqa_diamond", model: "injected-model" },
      sessionId: "dataset-error",
      modelLayer: layerSucceed(
        Model,
        Model.of({
          generate: () =>
            effectSucceed({
              completion: "Answer: A",
              message: { role: MessageRole.Assistant, content: "Answer: A" },
            }),
        })
      ),
      datasetLayer: layerSucceed(
        Dataset,
        Dataset.of({
          size: effectSucceed(1),
          stream: () => streamFail(datasetError),
        })
      ),
      epochs: 1,
      maxConcurrency: 1,
    });
    expect(Either.isLeft(datasetOutput)).toBe(true);
    if (Either.isRight(datasetOutput))
      throw new Error("expected dataset failure");
    expect(datasetOutput.left).toEqual({
      category: "dataset",
      code: "transient",
      retryable: true,
      systemic: false,
    });
    expect(JSON.stringify(datasetOutput.left)).not.toContain(secret);
  });

  it("classifies systemic model failures and preserves interruption", async () => {
    const systemic = await runBenchmarkDefinition({
      benchmark: chatBenchmark,
      benchmarkConfig: { benchmarkId: "gpqa_diamond", model: "injected-model" },
      sessionId: "systemic",
      modelLayer: layerSucceed(
        Model,
        Model.of({
          generate: () =>
            effectFail(new ModelError({ message: "secret", status: 401 })),
        })
      ),
      datasetLayer: datasetLayer(samples.slice(0, 1)),
      epochs: 1,
      maxConcurrency: 1,
    });
    expect(Either.isLeft(systemic)).toBe(true);
    if (Either.isRight(systemic)) throw new Error("expected systemic failure");
    expect(systemic.left).toMatchObject({
      category: "model",
      status: 401,
      retryable: false,
      systemic: true,
    });

    const controller = new AbortController();
    const interrupted = runBenchmarkDefinition({
      benchmark: chatBenchmark,
      benchmarkConfig: { benchmarkId: "gpqa_diamond", model: "injected-model" },
      sessionId: "interrupted",
      modelLayer: layerSucceed(
        Model,
        Model.of({
          generate: () =>
            sleep("1 hour").pipe(
              map(() => ({
                completion: "never",
                message: { role: MessageRole.Assistant, content: "never" },
              }))
            ),
        })
      ),
      datasetLayer: datasetLayer(samples.slice(0, 1)),
      epochs: 1,
      maxConcurrency: 1,
      abortSignal: controller.signal,
    });
    controller.abort();
    expect(
      await interrupted.then(
        () => "resolved",
        () => "rejected"
      )
    ).toBe("rejected");
  });

  it("executes with a supplied responses model layer", async () => {
    const generatedInputs: (readonly Record<string, unknown>[])[] = [];
    const responsesModelLayer = layerSucceed(
      ResponsesModel,
      ResponsesModel.of({
        generate: (input) => {
          generatedInputs.push(input);
          return effectSucceed({
            outputItems: [],
            functionCalls: [],
            text: "correct",
            generationTimeMs: 1,
          });
        },
      })
    );
    const benchmark: Benchmark<ResponsesTestConfig> = {
      id: "responses-test",
      temperature: 0,
      defaultEpochs: 1,
      makeDatasetLayer: () => datasetLayer(samples.slice(0, 1)),
      makeLayer: (input) => {
        const modelLayer =
          input.responsesModelLayer ??
          layerFail(new Error("responses model layer was not supplied"));
        const solverLayer = layerEffect(Solver)(
          gen(function* () {
            const model = yield* ResponsesModel;
            return Solver.of((state) =>
              model
                .generate([{ role: "user", content: state.sample.input }], {})
                .pipe(
                  map((turn) => ({
                    ...state,
                    output: {
                      completion: turn.text,
                      message: {
                        role: MessageRole.Assistant,
                        content: turn.text,
                      },
                    },
                    completed: true,
                  }))
                )
            );
          })
        ).pipe(layerProvide(modelLayer));
        return layerMergeAll(
          input.datasetLayer ?? datasetLayer(samples.slice(0, 1)),
          solverLayer,
          layerSucceed(
            Scorer,
            Scorer.of((_state, target) =>
              effectSucceed({
                value: ScoreValue.Correct,
                answer: target.text,
                explanation: "correct",
              })
            )
          )
        );
      },
    };

    const output = await runBenchmarkDefinition({
      benchmark,
      benchmarkConfig: {
        benchmarkId: "responses-test",
        model: "injected-responses-model",
        transport: "responses",
      },
      sessionId: "responses-session",
      responsesModelLayer,
      datasetLayer: datasetLayer(samples.slice(0, 1)),
      epochs: 1,
      maxConcurrency: 1,
    });

    expect(Either.isRight(output)).toBe(true);
    expect(generatedInputs).toEqual([
      [{ role: "user", content: "Question 0" }],
    ]);
  });
});
