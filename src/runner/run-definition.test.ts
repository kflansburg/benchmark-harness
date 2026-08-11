import { describe, expect, it } from "bun:test";

import { fromIterable } from "effect/Chunk";
import { gen, map, sleep, succeed as effectSucceed, sync } from "effect/Effect";
import type { Layer } from "effect/Layer";
import {
  effect as layerEffect,
  fail as layerFail,
  mergeAll as layerMergeAll,
  provide as layerProvide,
  succeed as layerSucceed,
} from "effect/Layer";
import { fromChunk } from "effect/Stream";

import type { GpqaBenchmarkConfig } from "../benchmarks/benchmark-config";
import { defineChatBenchmark } from "../benchmarks/define-chat-benchmark";
import type { Benchmark } from "../benchmarks/types";
import type { Sample } from "../harness/core";
import { MessageRole, ScoreValue } from "../harness/core";
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
    const benchmark = defineChatBenchmark<GpqaBenchmarkConfig>({
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

    const output = await runBenchmarkDefinition({
      benchmark,
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
    const benchmark: Benchmark = {
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
        benchmarkId: "gpqa_diamond",
        model: "injected-responses-model",
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
