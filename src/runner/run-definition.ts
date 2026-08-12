import { FetchHttpClient } from "@effect/platform";
import type { HttpClient } from "@effect/platform";
import { match as effectMatch, provide } from "effect/Effect";
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
import {
  DatasetError,
  isRetryableModelError,
  isSystemicModelError,
  ModelError,
  SolverError,
} from "../harness/core";
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

export interface BenchmarkExecutionError {
  readonly category: "model" | "dataset" | "solver" | "internal";
  readonly code?: string;
  readonly status?: number;
  readonly retryable: boolean;
  readonly systemic: boolean;
}

export function runBenchmarkDefinition<C extends BenchmarkConfig>(
  input: RunBenchmarkDefinitionInput<C>
): AsyncEither<RunBenchmarkOutput, BenchmarkExecutionError> {
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
    runBenchmark(runConfig).pipe(
      provide(layers),
      effectMatch({
        onFailure: (error) => Either.left(toBenchmarkExecutionError(error)),
        onSuccess: (result) => Either.right(result),
      })
    ),
    runOpts
  )
    .then((execution) => {
      if (Either.isLeft(execution)) {
        return execution;
      }
      const result = execution.right;
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
    .catch((error) => {
      if (input.abortSignal?.aborted === true) throw error;
      return Either.left(INTERNAL_EXECUTION_ERROR);
    });
}

const INTERNAL_EXECUTION_ERROR = {
  category: "internal",
  retryable: false,
  systemic: true,
} as const satisfies BenchmarkExecutionError;

function toBenchmarkExecutionError(error: unknown): BenchmarkExecutionError {
  if (error instanceof ModelError) {
    return {
      category: "model",
      ...(safeStatus(error.status) !== undefined && {
        status: safeStatus(error.status),
      }),
      ...(safeCode(error) !== undefined && { code: safeCode(error) }),
      retryable: isRetryableModelError(error),
      systemic: isSystemicModelError(error),
    };
  }
  if (error instanceof DatasetError) {
    const retryable = safeBoolean(error, "retryable") ?? false;
    return {
      category: "dataset",
      ...(safeCode(error) !== undefined && { code: safeCode(error) }),
      retryable,
      systemic: !retryable,
    };
  }
  if (error instanceof SolverError) {
    return { category: "solver", retryable: false, systemic: true };
  }
  return INTERNAL_EXECUTION_ERROR;
}

function safeCode(error: object): string | undefined {
  const value = "code" in error ? error.code : undefined;
  return typeof value === "string" && /^[a-z0-9_]{1,64}$/u.test(value)
    ? value
    : undefined;
}

function safeStatus(value: number | undefined): number | undefined {
  return Number.isSafeInteger(value) &&
    value !== undefined &&
    value >= 100 &&
    value <= 599
    ? value
    : undefined;
}

function safeBoolean(error: object, key: string): boolean | undefined {
  const value = key in error ? error[key as keyof typeof error] : undefined;
  return typeof value === "boolean" ? value : undefined;
}
