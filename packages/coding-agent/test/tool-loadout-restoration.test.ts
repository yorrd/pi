import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage, getCurrentTools, type TranscriptContext } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, test } from "vitest";
import type { ToolDefinition } from "../src/core/extensions/index.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createHarness, createTestUiContext } from "./suite/harness.ts";

function deferredTool(name: string): ToolDefinition {
	return {
		name,
		label: name,
		description: `${name} tool`,
		promptSnippet: `Use ${name}`,
		promptGuidelines: [`Use ${name} for its task.`],
		exposure: "deferred",
		parameters: Type.Object({}),
		execute: async () => ({ content: [{ type: "text", text: "ok" }], details: {} }),
	};
}

function recordedLoadout(names: string[]): SessionManager {
	const sessionManager = SessionManager.inMemory();
	sessionManager.appendMessage({
		role: "system",
		content: "",
		toolsAdded: names.map((name) => ({ name, description: `${name} tool`, parameters: Type.Object({}) })),
		timestamp: 0,
	});
	return sessionManager;
}

describe("tool loadout restoration", () => {
	test("resumes discovered tools in recorded order when an earlier tool registers during session_start", async () => {
		const harness = await createHarness({
			tools: [],
			sessionManager: recordedLoadout(["late_tool", "early_tool"]),
			extensionFactories: [
				(pi) => {
					pi.registerTool(deferredTool("early_tool"));
					pi.on("session_start", () => pi.registerTool(deferredTool("late_tool")));
				},
			],
		});
		try {
			await harness.session.bindExtensions({ uiContext: createTestUiContext() });
			expect(harness.session.getActiveToolNames()).toEqual(["late_tool", "early_tool"]);

			const requests: TranscriptContext[] = [];
			harness.setResponses([
				(context) => {
					requests.push(context);
					return fauxAssistantMessage("done");
				},
			]);
			await harness.session.prompt("continue");

			expect(requests).toHaveLength(1);
			expect(getCurrentTools(requests[0].messages).map((tool) => tool.name)).toEqual(["late_tool", "early_tool"]);
			const prompt = harness.session.systemPrompt;
			expect(prompt.indexOf("- late_tool: Use late_tool")).toBeLessThan(
				prompt.indexOf("- early_tool: Use early_tool"),
			);
		} finally {
			harness.cleanup();
		}
	});

	test("reloads without changing the prompt or executable loadout when registration order changes", async () => {
		const directory = mkdtempSync(join(tmpdir(), "pi-loadout-reload-"));
		let generation = 0;
		const resourceLoader = new DefaultResourceLoader({
			cwd: directory,
			agentDir: join(directory, "agent"),
			settingsManager: SettingsManager.inMemory(),
			extensionFactories: [
				(pi) => {
					generation++;
					const early = generation === 1 ? "first_tool" : "second_tool";
					const late = generation === 1 ? "second_tool" : "first_tool";
					pi.registerTool(deferredTool(early));
					pi.on("session_start", () => pi.registerTool(deferredTool(late)));
				},
			],
		});
		await resourceLoader.reload();
		const harness = await createHarness({
			tools: [],
			sessionManager: recordedLoadout(["first_tool", "second_tool"]),
			resourceLoader,
		});
		try {
			await harness.session.bindExtensions({ uiContext: createTestUiContext() });
			harness.setResponses([fauxAssistantMessage("done")]);
			await harness.session.prompt("before reload");
			const prompt = harness.session.systemPrompt;
			const declarations = harness.session.agent.state.tools.map(({ name, description, parameters }) => ({
				name,
				description,
				parameters,
			}));
			await harness.session.reload();

			expect(generation).toBe(2);
			expect(harness.session.getActiveToolNames()).toEqual(["first_tool", "second_tool"]);
			expect(harness.session.systemPrompt).toBe(prompt);
			expect(
				harness.session.agent.state.tools.map(({ name, description, parameters }) => ({
					name,
					description,
					parameters,
				})),
			).toEqual(declarations);
		} finally {
			harness.cleanup();
			rmSync(directory, { recursive: true, force: true });
		}
	});

	test("does not restore excluded tools or reactivate tools that were unavailable at the next run", async () => {
		let registerMissing = () => {};
		const harness = await createHarness({
			tools: [],
			sessionManager: recordedLoadout(["late_tool", "excluded_tool", "missing_tool", "early_tool"]),
			excludedToolNames: ["excluded_tool"],
			extensionFactories: [
				(pi) => {
					pi.registerTool(deferredTool("early_tool"));
					pi.registerTool(deferredTool("excluded_tool"));
					pi.on("session_start", () => pi.registerTool(deferredTool("late_tool")));
					registerMissing = () => pi.registerTool(deferredTool("missing_tool"));
				},
			],
		});
		try {
			await harness.session.bindExtensions({ uiContext: createTestUiContext() });
			expect(harness.session.getActiveToolNames()).toEqual(["late_tool", "early_tool"]);
			harness.setResponses([fauxAssistantMessage("done")]);
			await harness.session.prompt("continue without unavailable tools");
			registerMissing();

			expect(harness.session.getActiveToolNames()).toEqual(["late_tool", "early_tool"]);
			expect(harness.session.agent.state.tools.map((tool) => tool.name)).toEqual(["late_tool", "early_tool"]);
			harness.session.setActiveToolsByName(["late_tool", "early_tool", "excluded_tool", "missing_tool"]);
			expect(harness.session.getActiveToolNames()).toEqual(["late_tool", "early_tool", "missing_tool"]);
		} finally {
			harness.cleanup();
		}
	});

	test("an explicit empty loadout cancels restoration even when every tool is still pending", async () => {
		const harness = await createHarness({
			tools: [],
			sessionManager: recordedLoadout(["late_tool"]),
			extensionFactories: [
				(pi) => {
					pi.on("session_start", () => pi.registerTool(deferredTool("late_tool")));
				},
			],
		});
		try {
			harness.session.setActiveToolsByName([]);
			await harness.session.bindExtensions({ uiContext: createTestUiContext() });

			expect(harness.session.getActiveToolNames()).toEqual([]);
			expect(harness.session.agent.state.tools).toEqual([]);
		} finally {
			harness.cleanup();
		}
	});

	test("an explicit replacement loadout cancels pending restoration", async () => {
		const harness = await createHarness({
			tools: [],
			sessionManager: recordedLoadout(["late_tool", "early_tool"]),
			extensionFactories: [
				(pi) => {
					pi.registerTool(deferredTool("early_tool"));
					pi.registerTool(deferredTool("replacement_tool"));
					pi.on("session_start", () => pi.registerTool(deferredTool("late_tool")));
				},
			],
		});
		try {
			harness.session.setActiveToolsByName(["replacement_tool"]);
			await harness.session.bindExtensions({ uiContext: createTestUiContext() });

			expect(harness.session.getActiveToolNames()).toEqual(["replacement_tool"]);
		} finally {
			harness.cleanup();
		}
	});
});
