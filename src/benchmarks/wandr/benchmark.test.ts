import { describe, expect, it } from "bun:test";

import { FetchHttpClient } from "@effect/platform";
import { runPromise, scoped } from "effect/Effect";
import { build, provide } from "effect/Layer";

import { assertRight } from "../../internal/testing";
import { parseSchema } from "../../internal/zod";
import { WandrConfigSchema } from "../benchmark-config";
import { WANDR_BENCHMARK, wandrInferenceOverride } from "./benchmark";
describe("WANDR benchmark configuration", () => {
  it("forwards costTier into the solver inference override", () => {
    const parsed = parseSchema(WandrConfigSchema, {
      benchmarkId: "wandr",
      model: "openai/gpt-5.4",
      costTier: "high",
    });
    assertRight(parsed);
    expect(wandrInferenceOverride(parsed.right).costTier).toBe("high");
  });
  it("rejects a missing API key before constructing infrastructure", async () => {
    const parsed = parseSchema(WandrConfigSchema, {
      benchmarkId: "wandr",
      model: "openai/gpt-5.4",
    });
    assertRight(parsed);
    const layer = WANDR_BENCHMARK.makeLayer({
      benchmarkConfig: parsed.right,
      sessionId: "test-session",
    });
    const buildLayer = build(layer.pipe(provide(FetchHttpClient.layer))).pipe(
      scoped
    );

    await expect(runPromise(buildLayer)).rejects.toThrow(
      "wandr requires an API key"
    );
  });
});
