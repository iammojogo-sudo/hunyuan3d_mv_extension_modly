"use strict";
/* eslint-disable @typescript-eslint/no-require-imports */
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

const processor = async (input, params, context) => {
    if (!input.filePath)
        throw new Error("hunyuan3d2mv-texture: input.filePath is required");

    const extDir = __dirname;

    // Resolve venv Python — Windows vs Unix
    const isWin = process.platform === "win32";
    const pythonExe = isWin
        ? path.join(extDir, "venv", "Scripts", "python.exe")
        : path.join(extDir, "venv", "bin", "python");

    if (!fs.existsSync(pythonExe))
        throw new Error(
            `hunyuan3d2mv-texture: venv not found at ${pythonExe}. ` +
            "Please reinstall or repair the extension."
        );

    const workerScript = path.join(extDir, "texture_worker.py");

    // Models dir — prefer env var set by Modly, fall back to ~/.modly/models
    const modelsDir =
        process.env.MODELS_DIR ||
        path.join(require("os").homedir(), ".modly", "models");

    const workspaceDir = context.workspaceDir ||
        path.join(require("os").homedir(), ".modly", "workspace");

    // Serialise params for the worker
    const paramsJson = JSON.stringify(params || {});

    context.log(`Texture node starting — mesh: ${input.filePath}`);
    context.log(`Models dir: ${modelsDir}`);
    context.log(`Workspace dir: ${workspaceDir}`);
    context.progress(2, "Starting texture worker...");

    return new Promise((resolve, reject) => {
        const worker = spawn(pythonExe, [
            workerScript,
            input.filePath,
            paramsJson,
            modelsDir,
            workspaceDir,
        ], {
            env: {
                ...process.env,
                MODELS_DIR:    modelsDir,
                WORKSPACE_DIR: workspaceDir,
                EXTENSION_DIR: extDir,
            },
        });

        let outputPath = null;
        let errorBuf   = "";

        // Worker sends newline-delimited JSON on stdout
        let lineBuf = "";
        worker.stdout.on("data", (chunk) => {
            lineBuf += chunk.toString();
            const lines = lineBuf.split("\n");
            lineBuf = lines.pop(); // keep incomplete last line
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                try {
                    const msg = JSON.parse(trimmed);
                    if (msg.type === "progress") {
                        context.progress(msg.pct, msg.step || "");
                    } else if (msg.type === "log") {
                        context.log(msg.message || "");
                    } else if (msg.type === "done") {
                        outputPath = msg.output_path;
                    } else if (msg.type === "error") {
                        reject(new Error(msg.message || "Texture worker error"));
                    }
                } catch (_) {
                    // Non-JSON stdout line — log it
                    context.log(`[worker] ${trimmed}`);
                }
            }
        });

        // Stderr forwarded straight to context.log
        worker.stderr.on("data", (chunk) => {
            const text = chunk.toString().trim();
            if (text) context.log(`[worker stderr] ${text}`);
        });

        worker.on("error", (err) => {
            reject(new Error(`Failed to start texture worker: ${err.message}`));
        });

        worker.on("close", (code) => {
            if (outputPath) {
                context.log(`Texture complete: ${outputPath}`);
                resolve({ filePath: outputPath });
            } else if (code !== 0) {
                reject(new Error(
                    `Texture worker exited with code ${code}. ` +
                    (errorBuf ? `\n${errorBuf}` : "Check logs for details.")
                ));
            } else {
                reject(new Error("Texture worker finished but returned no output path."));
            }
        });
    });
};

module.exports = processor;
