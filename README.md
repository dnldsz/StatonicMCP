# Statonic MCP

A Model Context Protocol server that connects Claude to the Statonic video editor, enabling vision-based reference video analysis, AI-driven project authoring, and variation generation from clip library state.

## How it works

The server exposes four tools to Claude:

- `get_reference_frames` — reads a set of video frames sampled by the editor and returns them as base64 images for vision analysis. Claude identifies the structure of the video (hook, techniques, CTA) and writes the result back via `write_reference_result`, which the editor picks up via a file watcher and renders into an interactive slot assignment UI.
- `write_statonic_project` — writes a complete project JSON to the editor's watched load path. The editor detects the file change and loads the project immediately, allowing Claude to author or modify projects directly.
- `create_variations` — reads `variation-context.json` (written by the editor on session start, containing the current project and filtered clip library) and writes variation JSON files to the variations folder. Each file triggers a live update in the editor's variations panel.

## Stack

- `@modelcontextprotocol/sdk` for the MCP server transport
- Anthropic SDK for vision calls within `get_reference_frames`
- Node.js `fs` for file-based communication with the editor process
