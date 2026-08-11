import { FetchHttpClient } from "@effect/platform";
import type { HttpClient } from "@effect/platform";
import { provide } from "effect/Effect";
import type { Layer } from "effect/Layer";
import {
  mergeAll as layerMergeAll,
  provide as layerProvide,
  succeed as layerSucceed,
} from "effect/Layer";

import type {
  BenchmarkConfig,
  BenchmarkRunConfig,
} from "../benchmarks/benchmark-config";
import { modelFromConfig } from "../benchmarks/benchmark-config";
import type { Benchmark } from "../benchmarks/types";
import type { Dataset } from "../harness/dataset";
import type { Model } from "../harness/model";
import {
  CheckpointStore,
  NOOP_CHECKPOINT_STORE,
  NOOP_PROGRESS_REPORTER,
  ProgressReporter,
} from "../harness/progress";
import type { RunConfig } from "../harness/run";
import { runBenchmark } from "../harness/run";
import { runHarnessPromise } from "../internal/effect-logger";
import type { AsyncEither } from "../internal/either";
import { Either } from "../internal/either";
import { wLog } from "../internal/log";
import type { ResponsesModel } from "../providers/responses-model";
import type { ResultStoreService } from "../results/result-store";
import type { RunBenchmarkInput, RunBenchmarkOutput } from "./run-by-id";

export interface RunBenchmarkDefinitionInput<
  C extends BenchmarkConfig = BenchmarkRunConfig,
> extends Omit<
  RunBenchmarkInput,
  "benchmarkId" | "apiKey" | "benchmarkConfig" | "resultStore"
> {
  readonly benchmark: Benchmark<C>;
  readonly benchmarkConfig: C;
  readonly apiKey?: string;
  readonly modelLayer?: Layer<Model, Error, HttpClient.HttpClient>;
  readonly datasetLayer?: Layer<Dataset>;
  readonly responsesModelLayer?: Layer<
    ResponsesModel,
    Error,
    HttpClient.HttpClient
  >;
  readonly resultStore?: ResultStoreService<C>;
}

export function runBenchmarkDefinition<C extends BenchmarkConfig>(
  input: RunBenchmarkDefinitionInput<C>
): AsyncEither<RunBenchmarkOutput, string> {
  const { benchmark } = input;
  const maxRetries = input.benchmarkConfig.maxRetries;
  const benchmarkLayer = benchmark.makeLayer({
    benchmarkConfig: input.benchmarkConfig,
    sessionId: input.sessionId,
    ...(input.apiKey !== undefined && { apiKey: input.apiKey }),
    ...(input.baseUrl !== undefined && { baseUrl: input.baseUrl }),
    ...(input.datasetRetry !== undefined && {
      datasetRetry: input.datasetRetry,
    }),
    ...(maxRetries !== undefined && { modelRetry: { maxRetries } }),
    ...(input.maxOutputTokensCeiling !== undefined && {
      maxOutputTokensCeiling: input.maxOutputTokensCeiling,
    }),
    ...(input.modelLayer !== undefined && { modelLayer: input.modelLayer }),
    ...(input.datasetLayer !== undefined && {
      datasetLayer: input.datasetLayer,
    }),
    ...(input.responsesModelLayer !== undefined && {
      responsesModelLayer: input.responsesModelLayer,
    }),
  });
  const progressLayer = layerSucceed(
    ProgressReporter,
    input.progressReporter ?? NOOP_PROGRESS_REPORTER
  );
  const checkpointLayer = layerSucceed(
    CheckpointStore,
    input.checkpointStore ?? NOOP_CHECKPOINT_STORE
  );
  const model = modelFromConfig(input.benchmarkConfig);
  const runConfig: RunConfig = {
    epochs: input.epochs,
    maxConcurrency: input.maxConcurrency,
    ...(input.range !== undefined && { range: input.range }),
    ...(benchmark.degradeSolverErrors !== undefined && {
      degradeSolverErrors: benchmark.degradeSolverErrors,
    }),
    logAnnotations: {
      benchmark: benchmark.id,
      session_id: input.sessionId,
      ...(model !== undefined && { model }),
    },
  };
  const layers = layerMergeAll(
    benchmarkLayer.pipe(layerProvide(FetchHttpClient.layer)),
    progressLayer,
    checkpointLayer
  );
  const runOpts =
    input.abortSignal !== undefined ? { signal: input.abortSignal } : undefined;
  return runHarnessPromise(
    runBenchmark(runConfig).pipe(provide(layers)),
    runOpts
  )
    .then((result) => {
      if (input.resultStore !== undefined) {
        return runHarnessPromise(
          input.resultStore.write({
            result,
            benchmark,
            benchmarkConfig: input.benchmarkConfig,
            epochs: input.epochs,
            sessionId: input.sessionId,
          })
        )
          .then((resultsPath) => Either.right({ result, resultsPath }))
          .catch((storeErr) => {
            wLog("Failed to persist benchmark results", {
              error: String(storeErr),
            });
            return Either.right({ result, resultsPath: null });
          });
      }
      return Either.right({ result, resultsPath: null });
    })
    .catch((error) => Either.left(String(error)));
}
