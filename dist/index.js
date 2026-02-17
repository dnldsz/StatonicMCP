#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema, } from '@modelcontextprotocol/sdk/types.js';
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync, mkdirSync } from 'fs';
import Anthropic from '@anthropic-ai/sdk';
import { spawnSync } from 'child_process';
import { tmpdir } from 'os';
import { join, extname, basename } from 'path';
import { randomBytes } from 'crypto';
// ── Helpers ────────────────────────────────────────────────────────────────
function uid() { return randomBytes(4).toString('hex'); }
function readProject(path) {
    if (!existsSync(path))
        throw new Error(`Project file not found: ${path}`);
    return JSON.parse(readFileSync(path, 'utf-8'));
}
function saveProject(path, project) {
    writeFileSync(path, JSON.stringify(project, null, 2));
}
function findSegment(project, id) {
    for (const track of project.tracks) {
        const seg = track.segments.find(s => s.id === id);
        if (seg)
            return { seg, track };
    }
    return null;
}
function extractFrame(videoPath, timeSec) {
    const tmp = join(tmpdir(), `frame_${uid()}.jpg`);
    const r = spawnSync('ffmpeg', [
        '-ss', String(timeSec),
        '-i', videoPath,
        '-vframes', '1',
        '-q:v', '3',
        '-vf', 'scale=640:-1',
        tmp, '-y',
    ], { stdio: 'pipe' });
    if (r.status !== 0)
        throw new Error(`ffmpeg failed: ${r.stderr?.toString()}`);
    const data = readFileSync(tmp);
    spawnSync('rm', ['-f', tmp]);
    return data.toString('base64');
}
function getVideoInfo(videoPath) {
    const r = spawnSync('ffprobe', [
        '-v', 'quiet', '-print_format', 'json', '-show_streams', videoPath,
    ], { encoding: 'utf-8' });
    if (r.status !== 0)
        throw new Error(`ffprobe failed: ${r.stderr}`);
    const json = JSON.parse(r.stdout);
    const v = json.streams?.find((s) => s.codec_type === 'video');
    let width = v?.width ?? 1080;
    let height = v?.height ?? 1920;
    const durationSec = parseFloat(v?.duration ?? '0');
    // Mobile videos are often encoded in landscape with a rotation tag
    // ffprobe reports raw encoded dimensions; we need display dimensions
    const tagRotate = parseInt(v?.tags?.rotate ?? '0');
    const sideRotate = parseInt(v?.side_data_list?.[0]?.rotation ?? '0');
    const rotate = tagRotate || sideRotate;
    // If rotated 90° or 270°, swap width and height for display dimensions
    if (Math.abs(rotate % 180) === 90) {
        ;
        [width, height] = [height, width];
    }
    return { width, height, durationSec, rotation: rotate };
}
function summariseProject(project) {
    const lines = [
        `Project: "${project.name}"  canvas: ${project.canvas.width}×${project.canvas.height}`,
        '',
    ];
    for (const track of project.tracks) {
        lines.push(`Track [${track.id}] "${track.label}" (${track.type})`);
        for (const seg of track.segments) {
            const start = (seg.startUs / 1e6).toFixed(3);
            const dur = (seg.durationUs / 1e6).toFixed(3);
            if (seg.type === 'video') {
                const v = seg;
                lines.push(`  [${v.id}] VIDEO "${v.name}"  ${start}s → +${dur}s` +
                    `  pos=(${v.clipX.toFixed(3)},${v.clipY.toFixed(3)})` +
                    `  scale=${v.clipScale.toFixed(3)}` +
                    `  crop=(L${v.cropLeft.toFixed(3)} R${v.cropRight.toFixed(3)} T${v.cropTop.toFixed(3)} B${v.cropBottom.toFixed(3)})` +
                    `  src="${v.src}"`);
            }
            else {
                const t = seg;
                lines.push(`  [${t.id}] TEXT "${t.text}"  ${start}s → +${dur}s` +
                    `  pos=(${t.x.toFixed(3)},${t.y.toFixed(3)})` +
                    `  fontSize=${t.fontSize}  color=${t.color}` +
                    `  bold=${t.bold}  italic=${t.italic}`);
            }
        }
    }
    return lines.join('\n');
}
function extractKeyframes(videoPath, count) {
    const info = getVideoInfo(videoPath);
    const duration = info.durationSec;
    const timestamps = [];
    const frames = [];
    // Build filter for rotation correction
    // Rotation tag indicates how the video SHOULD be displayed
    // We need to apply that rotation to correct the encoded orientation
    const filters = [];
    if (info.rotation !== 0) {
        const absRotate = Math.abs(info.rotation % 360);
        if (absRotate === 90 || absRotate === 270) {
            // For -90° rotation tag: rotate 90° clockwise to display correctly
            // For +90° rotation tag: rotate 90° counter-clockwise to display correctly
            if ((info.rotation === -90) || (info.rotation === 270)) {
                filters.push('transpose=1'); // 90° clockwise
            }
            else if ((info.rotation === 90) || (info.rotation === -270)) {
                filters.push('transpose=2'); // 90° counter-clockwise
            }
        }
        else if (absRotate === 180) {
            filters.push('transpose=1,transpose=1'); // 180°
        }
    }
    filters.push('scale=640:-2');
    const vf = filters.join(',');
    // Extract frames evenly across the video
    for (let i = 0; i < count; i++) {
        const t = (duration / (count + 1)) * (i + 1);
        timestamps.push(t);
        const tmp = join(tmpdir(), `frame_${uid()}.jpg`);
        const args = [
            '-ss', String(t),
            '-i', videoPath,
            '-vf', vf,
            '-vframes', '1',
            '-q:v', '2',
            tmp, '-y',
        ];
        const r = spawnSync('ffmpeg', args, { stdio: 'pipe' });
        if (r.status === 0) {
            const data = readFileSync(tmp);
            frames.push(data.toString('base64'));
            spawnSync('rm', ['-f', tmp]);
        }
    }
    return { timestamps, frames };
}
// ── Tool definitions ───────────────────────────────────────────────────────
const TOOLS = [
    {
        name: 'read_project',
        description: 'Read a Statonic project JSON file. Returns a human-readable summary of all tracks and segments, plus the full JSON for reference. Use this first to understand the project before making edits.',
        inputSchema: {
            type: 'object',
            properties: {
                path: { type: 'string', description: 'Absolute path to the .json project file' },
            },
            required: ['path'],
        },
    },
    {
        name: 'update_segment',
        description: `Update one or more properties of an existing segment (video or text) and save.

Video segment writable properties:
  clipX, clipY        — canvas position (-1=left/bottom, 0=center, 1=right/top)
  clipScale           — scale (1.0 = fill canvas height, <1 = smaller, >1 = larger)
  scaleKeyframes      — array of {timeMs: number, scale: number} for zoom animations (use add_zoom_keyframes tool instead)
  cropLeft, cropRight, cropTop, cropBottom  — crop fractions 0–1
  startUs, durationUs — timeline position / length in microseconds
  sourceStartUs, sourceDurationUs — source trim in microseconds

Text segment writable properties:
  text, x, y, fontSize, color, bold, italic,
  strokeEnabled, strokeColor, textAlign, textScale, startUs, durationUs`,
        inputSchema: {
            type: 'object',
            properties: {
                path: { type: 'string', description: 'Absolute path to the .json project file' },
                id: { type: 'string', description: 'Segment ID (from read_project output)' },
                patch: {
                    type: 'object',
                    description: 'Key/value pairs to update on the segment (partial update)',
                },
            },
            required: ['path', 'id', 'patch'],
        },
    },
    {
        name: 'delete_segment',
        description: 'Delete a segment from the project and save.',
        inputSchema: {
            type: 'object',
            properties: {
                path: { type: 'string', description: 'Absolute path to the .json project file' },
                id: { type: 'string', description: 'Segment ID to delete' },
            },
            required: ['path', 'id'],
        },
    },
    {
        name: 'add_text_segment',
        description: `Add a new text overlay to the project and save. Reuses an existing text track if one exists, otherwise creates one.

Position reference (x, y):
  ( 0,  0) = canvas center
  (-1,  0) = left edge,  (1, 0) = right edge
  ( 0,  1) = top,        (0,-1) = bottom`,
        inputSchema: {
            type: 'object',
            properties: {
                path: { type: 'string', description: 'Absolute path to the .json project file' },
                text: { type: 'string', description: 'Text content' },
                start_sec: { type: 'number', description: 'Start time in seconds' },
                duration_sec: { type: 'number', description: 'Duration in seconds' },
                x: { type: 'number', description: 'Horizontal position -1 to 1 (default 0)' },
                y: { type: 'number', description: 'Vertical position -1 to 1 (default 0)' },
                font_size: { type: 'number', description: 'Font size in canvas px (default 80)' },
                color: { type: 'string', description: 'Hex color e.g. "#ffffff" (default white)' },
                bold: { type: 'boolean', description: 'Bold (default false)' },
                italic: { type: 'boolean', description: 'Italic (default false)' },
                stroke_enabled: { type: 'boolean', description: 'Stroke/outline (default false)' },
                stroke_color: { type: 'string', description: 'Stroke color hex (default "#000000")' },
                text_align: {
                    type: 'string', enum: ['left', 'center', 'right'],
                    description: 'Text alignment (default "center")',
                },
            },
            required: ['path', 'text', 'start_sec', 'duration_sec'],
        },
    },
    {
        name: 'get_frames',
        description: `Extract frames from a video file at specified timestamps and return them as images.
Use this to visually inspect video content. Claude can see and describe the returned frames.
Frames are scaled to 640px wide. Limit to ≤6 frames per call for speed.`,
        inputSchema: {
            type: 'object',
            properties: {
                video_path: { type: 'string', description: 'Absolute path to the video file' },
                times_sec: {
                    type: 'array',
                    items: { type: 'number' },
                    description: 'Timestamps in seconds to extract (max 6)',
                },
            },
            required: ['video_path', 'times_sec'],
        },
    },
    {
        name: 'get_video_info',
        description: 'Get width, height, and duration of a video file via ffprobe.',
        inputSchema: {
            type: 'object',
            properties: {
                video_path: { type: 'string', description: 'Absolute path to the video file' },
            },
            required: ['video_path'],
        },
    },
    {
        name: 'render_preview',
        description: `Render a composite preview frame of the project at a specific time — all video layers cropped and positioned, text overlays drawn on top. Returns a JPEG image so you can see exactly what the canvas looks like: where subjects are, where overlays sit, what's obstructed. Use this before repositioning elements so you can make informed placement decisions.`,
        inputSchema: {
            type: 'object',
            properties: {
                project_path: { type: 'string', description: 'Absolute path to the .json project file' },
                time_sec: { type: 'number', description: 'Time in seconds to render (default: 0.5s into the first active clip)' },
            },
            required: ['project_path'],
        },
    },
    {
        name: 'analyze_video_clip',
        description: `Extract keyframes from a video clip for analysis. Returns frames as images for Claude to analyze and generate metadata. After Claude provides analysis, saves metadata JSON file next to the video. Use this to understand what's in a clip before selecting it for a project.`,
        inputSchema: {
            type: 'object',
            properties: {
                video_path: { type: 'string', description: 'Absolute path to the video file' },
                category: { type: 'string', description: 'Category/subject area (e.g., "physics", "math", "generic"). Optional.' },
                keyframe_count: { type: 'number', description: 'Number of keyframes to extract for analysis (default: 4)' },
                metadata: {
                    type: 'object',
                    description: 'Optional: Claude\'s analysis to save as metadata. If provided, saves JSON file. Should include: description, tags, mood, subject_visible, subject_position, setting',
                },
            },
            required: ['video_path'],
        },
    },
    {
        name: 'index_clip_bank',
        description: `Scan a folder (recursively) and analyze all video clips, building a searchable index. Generates metadata for clips that don't have it yet. Creates/updates index.json in the folder root.`,
        inputSchema: {
            type: 'object',
            properties: {
                folder_path: { type: 'string', description: 'Absolute path to the clip bank folder' },
                regenerate: { type: 'boolean', description: 'Re-analyze clips even if metadata exists (default: false)' },
            },
            required: ['folder_path'],
        },
    },
    {
        name: 'search_clip_bank',
        description: `Search the clip bank for clips matching a description. Returns clip metadata for Claude to rank and select the best matches. Use this to find the best clip for a specific video segment.`,
        inputSchema: {
            type: 'object',
            properties: {
                index_path: { type: 'string', description: 'Absolute path to index.json (or folder containing it)' },
                query: { type: 'string', description: 'Description of what you\'re looking for (e.g., "focused student studying chemistry")' },
                category: { type: 'string', description: 'Filter by category (optional)' },
            },
            required: ['index_path', 'query'],
        },
    },
    {
        name: 'analyze_statonic_library',
        description: `Analyze clips in Statonic's app library (~/Library/Application Support/Statonic/clip-library/clips/). Returns unanalyzed clips for Claude to analyze. After analysis, call this again with metadata to save. Convenient way to analyze the app's managed clip library without specifying full paths.`,
        inputSchema: {
            type: 'object',
            properties: {
                clip_id: { type: 'string', description: 'Specific clip ID to analyze (optional - if omitted, returns all unanalyzed clips)' },
                metadata: {
                    type: 'object',
                    description: 'Analysis results to save (description, tags, mood, subject_visible, subject_position, setting)',
                },
                force: { type: 'boolean', description: 'If true, include already-analyzed clips (for re-analysis with updated prompts)' },
            },
        },
    },
    {
        name: 'search_statonic_library',
        description: `Search Statonic's analyzed clip library for clips matching a query. Returns all analyzed clips with their metadata for Claude to rank by relevance. Use this to find the best clips for a specific purpose.`,
        inputSchema: {
            type: 'object',
            properties: {
                query: { type: 'string', description: 'What you\'re looking for (e.g., "student studying chemistry", "person explaining physics")' },
                category: { type: 'string', description: 'Optional: filter by category (math, physics, chemistry, biology, coding, generic)' },
                account_id: { type: 'string', description: 'Optional: filter by account ID (daniel, stacy, etc.)' },
            },
            required: ['query'],
        },
    },
    {
        name: 'get_clips_by_category',
        description: `Get clips from Statonic library filtered by category (hook/gizmo/showcase). Automatically uses the currently active account in Statonic app. Returns full clip details including path, duration, and dimensions needed to build projects.`,
        inputSchema: {
            type: 'object',
            properties: {
                category: {
                    type: 'string',
                    enum: ['hook', 'gizmo', 'showcase'],
                    description: 'Clip category to filter by'
                },
            },
            required: ['category'],
        },
    },
    {
        name: 'write_statonic_project',
        description: `Write a Statonic project JSON file.

TEXT STYLING & POSITIONING RULES (follow these automatically):
1. Font size: 75-100px (use 85-90 for multi-line text, adjust based on text length)
2. Break text into 2-3 lines using \\n to prevent going off screen
3. Position: y = 0.25 to 0.31 (lower middle of top half), x = 0 (centered)
4. Coordinate system: x=0,y=0 is center; y=1 is top; y=-1 is bottom
5. After creating project, use render_preview to verify text doesn't block subjects
6. If blocking subject's face/upper body, adjust y down to 0.22-0.25

HOOK FORMULA EXAMPLES (proven high-performance):
- "how to [VERB] so fast it feels illegal" (CAPS on key verb, e.g. "how to LEARN\\nso fast it\\nfeels illegal")
- "99% of [audience] do X the WRONG way" (specific number creates credibility)
- "the [topic] trick nobody talks about"
- 2-3 lines, 4-7 words each — NOT one long sentence
- Use CAPS for ONE emphasis word per line, not the whole text
- Use generate_hook_options tool to get 5 high-quality hook variants before writing

Example text formatting:
- "how to study chemistry" → "how to study\\nchemistry and get\\n99/100" (fontSize: 90)
- "ACTIVE RECALL 🤫" → single line is fine (fontSize: 75-100)

IMPORTANT: VideoSegment requires: src (not sourceFilePath), name, fileDurationUs, sourceWidth, sourceHeight. Track requires: label. See example:

{
  "name": "My Video",
  "canvas": {"width": 1080, "height": 1920},
  "tracks": [
    {
      "id": "track-1",
      "label": "Base",
      "type": "video",
      "segments": [{
        "id": "seg-1",
        "type": "video",
        "src": "/path/to/video.mp4",
        "name": "clip-name",
        "startUs": 0,
        "durationUs": 4200000,
        "sourceStartUs": 0,
        "sourceDurationUs": 4200000,
        "fileDurationUs": 5000000,
        "sourceWidth": 1080,
        "sourceHeight": 1920,
        "clipX": 0, "clipY": 0, "clipScale": 1,
        "cropLeft": 0, "cropRight": 0, "cropTop": 0, "cropBottom": 0
      }]
    },
    {
      "id": "track-2",
      "label": "Text",
      "type": "text",
      "segments": [{
        "id": "text-1",
        "type": "text",
        "text": "Hello",
        "startUs": 0,
        "durationUs": 2000000,
        "x": 0, "y": 0,
        "fontSize": 80,
        "color": "#ffffff",
        "fontFamily": "Arial"
      }]
    }
  ]
}`,
        inputSchema: {
            type: 'object',
            properties: {
                project: {
                    type: 'object',
                    description: 'Complete Statonic project JSON'
                },
                filename: {
                    type: 'string',
                    description: 'Output filename (e.g., "biology-study-video.json")'
                },
            },
            required: ['project', 'filename'],
        },
    },
    {
        name: 'get_reference_frames',
        description: `Read the pending reference video analysis request and return each scene frame as an image for you to analyze.

After the user clicks "Copy Reference" in the editor and selects a video, call this tool to see the extracted frames. Then call write_reference_result with your analysis.`,
        inputSchema: { type: 'object', properties: {}, required: [] },
    },
    {
        name: 'write_reference_result',
        description: `Write your analysis of the reference video frames back to the editor. The editor modal will automatically populate with the detected slots.

Call this after get_reference_frames. For text that appears across MULTIPLE consecutive slots (e.g. hook text while background clips change, or "Students who follow me and use:" over all technique slots), put it in spanning_texts — NOT in each slot's detectedText. Each slot's detectedText should only contain the unique text for that slot (e.g. "GAMIFICATION", "PAST PAPERS").`,
        inputSchema: {
            type: 'object',
            properties: {
                slots: {
                    type: 'array',
                    description: 'Analyzed slots, one per scene. Leave detectedText empty for slots covered by a spanning_text.',
                    items: {
                        type: 'object',
                        properties: {
                            startSec: { type: 'number', description: 'Scene start time in seconds' },
                            durationSec: { type: 'number', description: 'Scene duration in seconds' },
                            thumbnailPath: { type: 'string', description: 'Path to the extracted frame image' },
                            detectedText: { type: 'string', description: 'The complete text overlay visible in this slot — ALL lines, use \\n for line breaks' },
                            clipType: { type: 'string', enum: ['hook', 'gizmo', 'showcase'], description: 'Type of clip' },
                            description: { type: 'string', description: 'Brief description of what is shown' },
                        },
                        required: ['startSec', 'durationSec', 'thumbnailPath', 'clipType'],
                    },
                },
                spanning_texts: {
                    type: 'array',
                    description: 'Text overlays that persist across multiple consecutive slots (e.g. hook text shown while background clips change).',
                    items: {
                        type: 'object',
                        properties: {
                            text: { type: 'string', description: 'The persistent text (use \\n for line breaks)' },
                            fromSlot: { type: 'number', description: '0-based index of first slot this text covers' },
                            toSlot: { type: 'number', description: '0-based index of last slot this text covers (inclusive)' },
                        },
                        required: ['text', 'fromSlot', 'toSlot'],
                    },
                },
                hookTextY: { type: 'number', description: 'Y position for hook text overlay. Range: 1=top, 0=center, -1=bottom. Measure from reference frames: y = 1 - 2*(pixels_from_top / frame_height).' },
                spanningTextY: { type: 'number', description: 'Y position for the spanning/persistent text (e.g. "Students who follow me and use:"). Must be ABOVE slotTextY (higher value). Measure from reference frames.' },
                slotTextY: { type: 'number', description: 'Y position for per-slot technique text (e.g. "GAMIFICATION"). Must be BELOW spanningTextY (lower value). Measure from reference frames.' },
            },
            required: ['slots'],
        },
    },
    {
        name: 'list_templates',
        description: 'List all available video structure templates. Returns id, name, description, and slot count for each template.',
        inputSchema: { type: 'object', properties: {}, required: [] },
    },
    {
        name: 'use_template',
        description: `Create a Statonic project from a template. Auto-selects clips by category for unfilled slots.

Workflow:
1. Call list_templates to see available templates
2. Get clips with get_clips_by_category to know what's available
3. Call use_template with slot overrides (clip_id + text per slot)
4. Verify with render_preview`,
        inputSchema: {
            type: 'object',
            properties: {
                template_id: { type: 'string', description: 'Template ID (from list_templates)' },
                project_name: { type: 'string', description: 'Project name (optional, defaults to template name + date)' },
                slots: {
                    type: 'array',
                    description: 'Array of slot overrides. Unfilled slots auto-select clips by category.',
                    items: {
                        type: 'object',
                        properties: {
                            slot_id: { type: 'string', description: 'Slot ID from the template (e.g. "hook", "technique_1")' },
                            clip_id: { type: 'string', description: 'Clip ID from the library (optional — auto-selected if omitted)' },
                            text: { type: 'string', description: 'Text overlay for this slot (optional — uses template example if omitted)' },
                        },
                        required: ['slot_id'],
                    },
                },
            },
            required: ['template_id'],
        },
    },
    {
        name: 'generate_hook_options',
        description: `Generate 5 high-quality hook text options for a given topic, using proven formulas from hook-knowledge.json.

Returns formatted hook texts ready to paste into write_statonic_project.
Call this before creating any project to get the best hook text.`,
        inputSchema: {
            type: 'object',
            properties: {
                topic: { type: 'string', description: 'The video topic (e.g. "chemistry study", "fitness motivation")' },
                count: { type: 'number', description: 'Number of options to generate (default 5)' },
            },
            required: ['topic'],
        },
    },
    {
        name: 'learn_from_hook_video',
        description: `Analyze a trending video file to extract its hook formula and text. Appends the learned example to hook-knowledge.json for future use.

Use this to build up your hook knowledge base from high-performing videos.`,
        inputSchema: {
            type: 'object',
            properties: {
                video_path: { type: 'string', description: 'Absolute path to the video file to analyze' },
            },
            required: ['video_path'],
        },
    },
    {
        name: 'add_zoom_keyframes',
        description: `Add scale/zoom animation keyframes to a video segment. Creates smooth zoom in/out effects.

Examples:
  • "zoom in 20% for the hook" - adds keyframes to scale from 1.0 to 1.2 over the segment
  • "zoom out 10% from 2s to 4s" - adds keyframes at specific times
  • "subtle push in" - gentle 1.0 to 1.15 scale animation

The zoom animation interpolates linearly between keyframes. Keyframes are relative to the segment's start time.`,
        inputSchema: {
            type: 'object',
            properties: {
                path: { type: 'string', description: 'Absolute path to the .json project file' },
                segment_id: { type: 'string', description: 'Video segment ID to add zoom to' },
                keyframes: {
                    type: 'array',
                    description: 'Array of {time_sec, scale} keyframes. time_sec is relative to segment start (0 = segment start)',
                    items: {
                        type: 'object',
                        properties: {
                            time_sec: { type: 'number', description: 'Time in seconds from segment start' },
                            scale: { type: 'number', description: 'Scale value (1.0 = fill canvas height, 1.2 = 20% zoom in, 0.8 = zoom out)' },
                        },
                        required: ['time_sec', 'scale'],
                    },
                },
            },
            required: ['path', 'segment_id', 'keyframes'],
        },
    },
    {
        name: 'get_suitable_audio',
        description: `Find audio tracks suitable for a video project based on timing requirements. Returns audios where the drop point occurs after the hook ends and the audio is long enough to cover the entire video. Calculates exact audio positioning so the drop hits at the transition point. Use this when creating projects that need audio synced to clip transitions.`,
        inputSchema: {
            type: 'object',
            properties: {
                hook_duration_sec: {
                    type: 'number',
                    description: 'Hook duration in seconds (e.g., 4.2). The audio drop must occur after this time.'
                },
                total_duration_sec: {
                    type: 'number',
                    description: 'Total video duration in seconds (e.g., 6.4). The audio must be at least this long.'
                },
                prefer_closest: {
                    type: 'boolean',
                    description: 'If true, prefer audio with drop time closest to hook duration (default: false, picks randomly)'
                }
            },
            required: ['hook_duration_sec', 'total_duration_sec']
        }
    },
];
// ── Server ─────────────────────────────────────────────────────────────────
const server = new Server({ name: 'iterate-mcp', version: '0.1.0' }, { capabilities: { tools: {} } });
server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args = {} } = request.params;
    try {
        switch (name) {
            case 'read_project': {
                const project = readProject(args['path']);
                return {
                    content: [
                        {
                            type: 'text',
                            text: summariseProject(project) + '\n\n--- Full JSON ---\n' + JSON.stringify(project, null, 2),
                        },
                    ],
                };
            }
            case 'update_segment': {
                const { path, id, patch } = args;
                const project = readProject(path);
                const found = findSegment(project, id);
                if (!found)
                    throw new Error(`Segment "${id}" not found`);
                const { seg, track } = found;
                const idx = track.segments.indexOf(seg);
                track.segments[idx] = { ...seg, ...patch };
                saveProject(path, project);
                return { content: [{ type: 'text', text: `Updated segment "${id}". Open the project in Statonic (File → Open) to see the changes.` }] };
            }
            case 'delete_segment': {
                const { path, id } = args;
                const project = readProject(path);
                let deleted = false;
                for (const track of project.tracks) {
                    const idx = track.segments.findIndex(s => s.id === id);
                    if (idx !== -1) {
                        track.segments.splice(idx, 1);
                        deleted = true;
                        break;
                    }
                }
                if (!deleted)
                    throw new Error(`Segment "${id}" not found`);
                saveProject(path, project);
                return { content: [{ type: 'text', text: `Deleted segment "${id}".` }] };
            }
            case 'add_text_segment': {
                const { path, text, start_sec, duration_sec, x = 0, y = 0, font_size = 80, color = '#ffffff', bold = false, italic = false, stroke_enabled = false, stroke_color = '#000000', text_align = 'center', } = args;
                const project = readProject(path);
                let textTrack = project.tracks.find(t => t.type === 'text');
                if (!textTrack) {
                    textTrack = { id: uid(), type: 'text', label: 'TEXT', segments: [] };
                    project.tracks.push(textTrack);
                }
                const seg = {
                    id: uid(), type: 'text', text,
                    startUs: Math.round(start_sec * 1e6),
                    durationUs: Math.round(duration_sec * 1e6),
                    x, y, fontSize: font_size, color, bold, italic,
                    strokeEnabled: stroke_enabled, strokeColor: stroke_color,
                    textAlign: text_align, textScale: 1,
                };
                textTrack.segments.push(seg);
                saveProject(path, project);
                return {
                    content: [{
                            type: 'text',
                            text: `Added text "${text}" (id: ${seg.id}) at ${start_sec}s for ${duration_sec}s at position (${x}, ${y}).`,
                        }],
                };
            }
            case 'add_zoom_keyframes': {
                const { path, segment_id, keyframes } = args;
                const project = readProject(path);
                const found = findSegment(project, segment_id);
                if (!found)
                    throw new Error(`Segment "${segment_id}" not found`);
                const { seg, track } = found;
                if (seg.type !== 'video')
                    throw new Error(`Segment "${segment_id}" is not a video segment`);
                const videoSeg = seg;
                const scaleKeyframes = keyframes.map(kf => ({
                    timeMs: kf.time_sec * 1000,
                    scale: kf.scale,
                }));
                // Update segment with keyframes
                const idx = track.segments.indexOf(seg);
                track.segments[idx] = { ...videoSeg, scaleKeyframes };
                saveProject(path, project);
                const kfDesc = keyframes.map(kf => `${kf.time_sec}s: ${kf.scale}x`).join(', ');
                return {
                    content: [{
                            type: 'text',
                            text: `Added zoom keyframes to segment "${segment_id}":\n${kfDesc}\n\nOpen the project in Statonic to preview the zoom animation.`,
                        }],
                };
            }
            case 'get_frames': {
                const { video_path, times_sec } = args;
                const frames = times_sec.slice(0, 6);
                const content = [];
                for (const t of frames) {
                    try {
                        const b64 = extractFrame(video_path, t);
                        content.push({ type: 'image', data: b64, mimeType: 'image/jpeg' });
                        content.push({ type: 'text', text: `↑ ${t.toFixed(2)}s` });
                    }
                    catch (e) {
                        content.push({ type: 'text', text: `Failed at ${t}s: ${e.message}` });
                    }
                }
                return { content };
            }
            case 'get_video_info': {
                const info = getVideoInfo(args['video_path']);
                return {
                    content: [{
                            type: 'text',
                            text: `Width: ${info.width}px\nHeight: ${info.height}px\nDuration: ${info.durationSec.toFixed(3)}s`,
                        }],
                };
            }
            case 'render_preview': {
                const { project_path } = args;
                const project = readProject(project_path);
                const { canvas, tracks } = project;
                const activeVideo = [];
                const activeText = [];
                // Default time: 0.5s into first video clip, or 0
                let time_sec = args.time_sec;
                if (time_sec === undefined) {
                    const firstVid = tracks.flatMap(t => t.segments).find(s => s.type === 'video');
                    time_sec = firstVid ? firstVid.startUs / 1e6 + 0.5 : 0;
                }
                for (let ti = 0; ti < tracks.length; ti++) {
                    for (const seg of tracks[ti].segments) {
                        const start = seg.startUs / 1e6;
                        const end = (seg.startUs + seg.durationUs) / 1e6;
                        if (time_sec < start || time_sec >= end)
                            continue;
                        if (seg.type === 'video')
                            activeVideo.push({ seg: seg, trackIdx: ti });
                        else if (seg.type === 'text')
                            activeText.push(seg);
                    }
                }
                activeVideo.sort((a, b) => a.trackIdx - b.trackIdx);
                // Build ffmpeg inputs: [0] = black canvas, [1..N] = video segments seeked to frame
                const ffArgs = [
                    '-f', 'lavfi', '-i', `color=c=black:s=${canvas.width}x${canvas.height}:r=30`,
                ];
                for (const { seg } of activeVideo) {
                    const seekTime = Math.max(0, seg.sourceStartUs / 1e6 + (time_sec - seg.startUs / 1e6));
                    ffArgs.push('-ss', String(seekTime), '-i', seg.src);
                }
                // Build filter_complex
                const fp = [];
                // Step 1: crop + scale each video to its visible display size
                for (let i = 0; i < activeVideo.length; i++) {
                    const { seg } = activeVideo[i];
                    const clipScale = seg.clipScale ?? 1;
                    const srcW = seg.sourceWidth ?? canvas.width;
                    const srcH = seg.sourceHeight ?? canvas.height;
                    const cropL = seg.cropLeft ?? 0, cropR = seg.cropRight ?? 0;
                    const cropT = seg.cropTop ?? 0, cropB = seg.cropBottom ?? 0;
                    const cW = Math.max(0.01, 1 - cropL - cropR);
                    const cH = Math.max(0.01, 1 - cropT - cropB);
                    const fullH = Math.round(clipScale * canvas.height / 2) * 2;
                    const fullW = Math.round((srcW / srcH) * fullH / 2) * 2;
                    const visW = Math.max(2, Math.round(fullW * cW / 2) * 2);
                    const visH = Math.max(2, Math.round(fullH * cH / 2) * 2);
                    const cropFilter = (cropL > 0 || cropR > 0 || cropT > 0 || cropB > 0)
                        ? `crop=iw*${cW}:ih*${cH}:iw*${cropL}:ih*${cropT},` : '';
                    fp.push(`[${i + 1}:v]${cropFilter}scale=${visW}:${visH}[sv${i}]`);
                }
                // Step 2: chain overlay filters from black canvas
                let cur = '[0:v]';
                for (let i = 0; i < activeVideo.length; i++) {
                    const { seg } = activeVideo[i];
                    const clipScale = seg.clipScale ?? 1;
                    const srcW = seg.sourceWidth ?? canvas.width, srcH = seg.sourceHeight ?? canvas.height;
                    const cropL = seg.cropLeft ?? 0, cropT = seg.cropTop ?? 0;
                    const fullH = Math.round(clipScale * canvas.height / 2) * 2;
                    const fullW = Math.round((srcW / srcH) * fullH / 2) * 2;
                    const x = Math.round((seg.clipX + 1) / 2 * canvas.width - fullW / 2 + cropL * fullW);
                    const y = Math.round((1 - seg.clipY) / 2 * canvas.height - fullH / 2 + cropT * fullH);
                    const out = `[ov${i}]`;
                    fp.push(`${cur}[sv${i}]overlay=${x}:${y}${out}`);
                    cur = out;
                }
                // Step 3: drawtext for each active text segment
                for (let i = 0; i < activeText.length; i++) {
                    const t = activeText[i];
                    const px = Math.round((t.x + 1) / 2 * canvas.width);
                    const py = Math.round((1 - t.y) / 2 * canvas.height);
                    const fs = Math.round(t.fontSize * (t.textScale ?? 1));
                    const col = t.color.replace('#', '0x') + 'ff';
                    // Escape characters special to ffmpeg drawtext
                    const esc = t.text
                        .replace(/\\/g, '\\\\')
                        .replace(/'/g, "\\'")
                        .replace(/:/g, '\\:')
                        .replace(/%/g, '%%');
                    const xExpr = t.textAlign === 'left' ? `${px}` : t.textAlign === 'right' ? `${px}-tw` : `${px}-tw/2`;
                    let dt = `drawtext=text='${esc}':fontsize=${fs}:fontcolor=${col}:x=${xExpr}:y=${py}-th/2`;
                    // Note: ffmpeg drawtext doesn't support 'bold' option - would need to use a bold font file instead
                    if (t.strokeEnabled) {
                        const bw = Math.max(1, Math.round(fs * (6.9 / 97.0)));
                        dt += `:bordercolor=${t.strokeColor.replace('#', '0x')}ff:borderw=${bw}`;
                    }
                    const out = i === activeText.length - 1 ? '[txtout]' : `[txt${i}]`;
                    fp.push(`${cur}${dt}${out}`);
                    cur = out;
                }
                // Step 4: scale down to 540px wide for a manageable image size
                fp.push(`${cur}scale=540:-2[out]`);
                const tmp = join(tmpdir(), `preview_${uid()}.jpg`);
                const r = spawnSync('ffmpeg', [
                    '-y', ...ffArgs,
                    '-filter_complex', fp.join(';'),
                    '-map', '[out]',
                    '-vframes', '1',
                    '-q:v', '2',
                    tmp,
                ], { stdio: 'pipe' });
                if (r.status !== 0) {
                    throw new Error(`ffmpeg failed:\n${r.stderr?.toString().slice(-800)}`);
                }
                const imgData = readFileSync(tmp);
                spawnSync('rm', ['-f', tmp]);
                return {
                    content: [
                        { type: 'image', data: imgData.toString('base64'), mimeType: 'image/jpeg' },
                        {
                            type: 'text',
                            text: `Composite at ${time_sec.toFixed(2)}s — canvas ${canvas.width}×${canvas.height}, preview scaled to 540px wide.`,
                        },
                    ],
                };
            }
            case 'analyze_video_clip': {
                const { video_path, category = 'general', keyframe_count = 4, metadata: providedMetadata } = args;
                const info = getVideoInfo(video_path);
                // If metadata is provided, save it
                if (providedMetadata) {
                    const fullMetadata = {
                        id: uid(),
                        path: video_path,
                        name: basename(video_path, extname(video_path)),
                        category: providedMetadata.category ?? category,
                        duration: info.durationSec,
                        width: info.width,
                        height: info.height,
                        description: providedMetadata.description ?? '',
                        tags: providedMetadata.tags ?? [],
                        mood: providedMetadata.mood ?? 'neutral',
                        subject_visible: providedMetadata.subject_visible ?? false,
                        subject_position: providedMetadata.subject_position ?? 'unknown',
                        setting: providedMetadata.setting ?? 'unknown',
                        keyframe_timestamps: providedMetadata.keyframe_timestamps ?? [],
                        added: new Date().toISOString(),
                        analyzed_by: 'claude-sonnet-4-5',
                    };
                    const metadataPath = video_path.replace(extname(video_path), '.json');
                    writeFileSync(metadataPath, JSON.stringify(fullMetadata, null, 2));
                    return {
                        content: [{
                                type: 'text',
                                text: `Saved metadata for "${fullMetadata.name}" to ${metadataPath}`,
                            }],
                    };
                }
                // Extract frames for analysis
                const { timestamps, frames } = extractKeyframes(video_path, keyframe_count);
                if (frames.length === 0) {
                    throw new Error('Failed to extract keyframes from video');
                }
                const content = [];
                // Return frames as images for Claude to analyze
                for (let i = 0; i < frames.length; i++) {
                    content.push({
                        type: 'image',
                        data: frames[i],
                        mimeType: 'image/jpeg',
                    });
                    content.push({
                        type: 'text',
                        text: `↑ Frame at ${timestamps[i].toFixed(1)}s`,
                    });
                }
                content.push({
                    type: 'text',
                    text: `\nVideo: ${basename(video_path)}\n` +
                        `Duration: ${info.durationSec.toFixed(1)}s | Size: ${info.width}×${info.height}\n\n` +
                        `Please analyze these keyframes and provide:\n\n` +
                        `1. **Description**: Detailed 2-3 sentence description (subject, action, setting, mood, visual elements)\n\n` +
                        `2. **Category**: Classify as ONE of these clip types:\n` +
                        `   - "hook": Student sitting at desk, ENTIRELY visible in frame, looking frustrated/stressed/overwhelmed\n` +
                        `   - "gizmo": Quiz/question interface visible on screen (Gizmo app = active recall learning)\n` +
                        `   - "showcase": All other content (textbook studying, overhead views, explanations, etc.)\n\n` +
                        `3. **Tags**: 5-10 relevant keywords\n\n` +
                        `4. **Mood**: Single word (focused, energetic, calm, frustrated, stressed, etc.)\n\n` +
                        `5. **Subject Visible**: true/false\n\n` +
                        `6. **Subject Position**: center, left, right, center-left, etc.\n\n` +
                        `7. **Setting**: indoor-desk, outdoor-park, classroom, lab, screen-recording, etc.\n\n` +
                        `After analyzing, call analyze_video_clip again with the metadata parameter to save it.`,
                });
                return { content };
            }
            case 'index_clip_bank': {
                const { folder_path, regenerate = false } = args;
                const videoExts = ['.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'];
                const clips = [];
                const categories = new Set();
                function scanFolder(dir, relativeCategory = '') {
                    const entries = readdirSync(dir);
                    for (const entry of entries) {
                        const fullPath = join(dir, entry);
                        const stat = statSync(fullPath);
                        if (stat.isDirectory()) {
                            scanFolder(fullPath, relativeCategory ? `${relativeCategory}/${entry}` : entry);
                        }
                        else if (videoExts.includes(extname(entry).toLowerCase())) {
                            const metadataPath = fullPath.replace(extname(fullPath), '.json');
                            let metadata = null;
                            if (!regenerate && existsSync(metadataPath)) {
                                try {
                                    metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'));
                                }
                                catch {
                                    // Invalid JSON, will regenerate
                                }
                            }
                            if (!metadata) {
                                // Create placeholder metadata - will be analyzed later
                                const info = getVideoInfo(fullPath);
                                metadata = {
                                    id: uid(),
                                    path: fullPath,
                                    name: basename(fullPath, extname(fullPath)),
                                    category: relativeCategory || 'general',
                                    duration: info.durationSec,
                                    width: info.width,
                                    height: info.height,
                                    description: '(pending analysis)',
                                    tags: [],
                                    mood: 'unknown',
                                    subject_visible: false,
                                    subject_position: 'unknown',
                                    setting: 'unknown',
                                    keyframe_timestamps: [],
                                    added: new Date().toISOString(),
                                    analyzed_by: 'pending',
                                };
                                writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
                            }
                            if (metadata) {
                                clips.push(metadata);
                                categories.add(metadata.category);
                            }
                        }
                    }
                }
                scanFolder(folder_path);
                const index = {
                    clips,
                    categories: Array.from(categories),
                    last_updated: new Date().toISOString(),
                };
                const indexPath = join(folder_path, 'index.json');
                writeFileSync(indexPath, JSON.stringify(index, null, 2));
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Indexed ${clips.length} clips across ${categories.size} categories.\n\n` +
                                `Categories: ${Array.from(categories).join(', ')}\n` +
                                `Index saved to: ${indexPath}\n\n` +
                                `Note: Some clips may need full analysis. Use analyze_video_clip on individual clips for detailed metadata.`,
                        },
                    ],
                };
            }
            case 'search_clip_bank': {
                const { index_path, query, category } = args;
                let indexFile = index_path;
                if (!index_path.endsWith('.json')) {
                    indexFile = join(index_path, 'index.json');
                }
                if (!existsSync(indexFile)) {
                    throw new Error(`Index file not found: ${indexFile}. Run index_clip_bank first.`);
                }
                const index = JSON.parse(readFileSync(indexFile, 'utf-8'));
                let candidates = index.clips;
                if (category) {
                    candidates = candidates.filter((c) => c.category === category);
                }
                // Return clip metadata for Claude to rank
                const clipSummaries = candidates.map((c, i) => `[${i}] **${c.name}** (${c.category})\n` +
                    `    Path: ${c.path}\n` +
                    `    Description: ${c.description}\n` +
                    `    Tags: ${c.tags.join(', ')}\n` +
                    `    Mood: ${c.mood} | Subject: ${c.subject_visible ? c.subject_position : 'none'} | Setting: ${c.setting}\n` +
                    `    Duration: ${c.duration.toFixed(1)}s | Size: ${c.width}×${c.height}`).join('\n\n');
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Query: "${query}"\n` +
                                `Found ${candidates.length} clips${category ? ` in category "${category}"` : ''}:\n\n` +
                                `${clipSummaries}\n\n` +
                                `Please rank these clips by relevance to the query and return the top 5 matches with:\n` +
                                `- Clip name and index\n` +
                                `- Relevance score (0-10)\n` +
                                `- Brief reasoning why it matches`,
                        },
                    ],
                };
            }
            case 'search_statonic_library': {
                const { query, category, account_id } = args;
                // Determine library path - try common locations
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const basePaths = [
                    join(homeDir, 'Library', 'Application Support', 'Statonic', 'clip-library'), // macOS
                    join(homeDir, 'AppData', 'Roaming', 'Statonic', 'clip-library'), // Windows
                    join(homeDir, '.config', 'Statonic', 'clip-library'), // Linux
                ];
                let basePath = '';
                for (const path of basePaths) {
                    if (existsSync(path)) {
                        basePath = path;
                        break;
                    }
                }
                if (!basePath) {
                    throw new Error('Statonic clip library not found. Make sure the app has been run and clips have been imported.');
                }
                // Collect all analyzed clips from all accounts or specific account
                const clips = [];
                const accountsPath = join(basePath, 'accounts');
                if (existsSync(accountsPath)) {
                    const accounts = readdirSync(accountsPath);
                    for (const accId of accounts) {
                        // Skip if filtering by account and this isn't it
                        if (account_id && accId !== account_id)
                            continue;
                        const clipsPath = join(accountsPath, accId, 'clips');
                        if (!existsSync(clipsPath))
                            continue;
                        const clipDirs = readdirSync(clipsPath);
                        for (const clipId of clipDirs) {
                            const clipDir = join(clipsPath, clipId);
                            try {
                                const stat = statSync(clipDir);
                                if (!stat.isDirectory())
                                    continue;
                                const metadataPath = join(clipDir, 'metadata.json');
                                if (existsSync(metadataPath)) {
                                    const meta = JSON.parse(readFileSync(metadataPath, 'utf-8'));
                                    // Only include analyzed clips
                                    if (meta.analyzed) {
                                        // Filter by category if specified
                                        if (!category || meta.category === category) {
                                            clips.push({
                                                id: meta.id,
                                                accountId: meta.accountId || accId,
                                                name: meta.name,
                                                path: meta.path,
                                                category: meta.category || 'uncategorized',
                                                description: meta.description || '',
                                                tags: meta.tags || [],
                                                mood: meta.mood || '',
                                                subject_visible: meta.subject_visible || false,
                                                subject_position: meta.subject_position || '',
                                                setting: meta.setting || '',
                                                duration: meta.duration,
                                                width: meta.width,
                                                height: meta.height,
                                            });
                                        }
                                    }
                                }
                            }
                            catch {
                                // Skip invalid clips
                            }
                        }
                    }
                }
                if (clips.length === 0) {
                    return {
                        content: [{
                                type: 'text',
                                text: 'No analyzed clips found in the library. Import and analyze some clips first.',
                            }],
                    };
                }
                // Format clips for Claude to rank
                const clipSummaries = clips.map((c, i) => `[${i}] **${c.name}** (${c.category}${c.accountId ? ` | ${c.accountId}` : ''})\n` +
                    `    Description: ${c.description}\n` +
                    `    Tags: ${c.tags.join(', ')}\n` +
                    `    Mood: ${c.mood} | Subject: ${c.subject_visible ? c.subject_position : 'none'} | Setting: ${c.setting}\n` +
                    `    Duration: ${c.duration.toFixed(1)}s | Size: ${c.width}×${c.height}\n` +
                    `    Path: ${c.path}`).join('\n\n');
                return {
                    content: [{
                            type: 'text',
                            text: `Query: "${query}"\n` +
                                `Found ${clips.length} analyzed clip${clips.length !== 1 ? 's' : ''}${category ? ` in category "${category}"` : ''}:\n\n` +
                                `${clipSummaries}\n\n` +
                                `Please rank these clips by relevance to the query "${query}" and return:\n` +
                                `1. Top 5 most relevant clips with:\n` +
                                `   - Clip index and name\n` +
                                `   - Relevance score (0-10)\n` +
                                `   - Brief reasoning why it matches\n` +
                                `2. If asking for specific information (like paths), provide that directly.`,
                        }],
                };
            }
            case 'get_clips_by_category': {
                const { category } = args;
                // Read current account from Statonic state file
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const stateFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'current-state.json');
                let currentAccountId = null;
                if (existsSync(stateFile)) {
                    try {
                        const state = JSON.parse(readFileSync(stateFile, 'utf-8'));
                        currentAccountId = state.currentAccountId;
                    }
                    catch {
                        // Ignore parse errors
                    }
                }
                if (!currentAccountId) {
                    return {
                        content: [{
                                type: 'text',
                                text: 'No account is currently active in Statonic. Please select an account in the app first.',
                            }],
                    };
                }
                // Get clip library path
                const libraryBasePaths = [
                    join(homeDir, 'Library', 'Application Support', 'Statonic', 'clip-library'), // macOS
                    join(homeDir, 'AppData', 'Roaming', 'Statonic', 'clip-library'), // Windows
                    join(homeDir, '.config', 'Statonic', 'clip-library'), // Linux
                ];
                let libraryBasePath = '';
                for (const path of libraryBasePaths) {
                    if (existsSync(path)) {
                        libraryBasePath = path;
                        break;
                    }
                }
                if (!libraryBasePath) {
                    throw new Error('Statonic clip library not found. Make sure the app has been run and clips have been imported.');
                }
                // Collect clips for current account with specified category
                const clips = [];
                const accountClipsPath = join(libraryBasePath, 'accounts', currentAccountId, 'clips');
                if (existsSync(accountClipsPath)) {
                    const clipDirs = readdirSync(accountClipsPath);
                    for (const clipId of clipDirs) {
                        const clipDir = join(accountClipsPath, clipId);
                        try {
                            const stat = statSync(clipDir);
                            if (!stat.isDirectory())
                                continue;
                            const metadataPath = join(clipDir, 'metadata.json');
                            if (existsSync(metadataPath)) {
                                const meta = JSON.parse(readFileSync(metadataPath, 'utf-8'));
                                // Only include analyzed clips with matching category
                                if (meta.analyzed && meta.category === category) {
                                    clips.push({
                                        id: meta.id,
                                        name: meta.name,
                                        path: meta.path,
                                        description: meta.description || '',
                                        tags: meta.tags || [],
                                        mood: meta.mood || '',
                                        duration: meta.duration,
                                        width: meta.width,
                                        height: meta.height,
                                    });
                                }
                            }
                        }
                        catch {
                            // Skip invalid clips
                        }
                    }
                }
                if (clips.length === 0) {
                    return {
                        content: [{
                                type: 'text',
                                text: `No ${category} clips found for account "${currentAccountId}". Import and analyze some clips first.`,
                            }],
                    };
                }
                // Write filter request for Electron app to pick up
                const filterFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'filter-request.json');
                writeFileSync(filterFile, JSON.stringify({
                    category,
                    accountId: currentAccountId,
                    requestedAt: new Date().toISOString()
                }, null, 2));
                // Format clips with full technical details for project building
                const clipList = clips.map((c, i) => `${i + 1}. **${c.name}** (ID: ${c.id})\n` +
                    `   Description: ${c.description}\n` +
                    `   Tags: ${c.tags.join(', ')}\n` +
                    `   For Statonic project:\n` +
                    `   - src: "${c.path}"\n` +
                    `   - name: "${c.name}"\n` +
                    `   - fileDurationUs: ${Math.round(c.duration * 1e6)}\n` +
                    `   - sourceWidth: ${c.width}\n` +
                    `   - sourceHeight: ${c.height}`).join('\n\n');
                return {
                    content: [{
                            type: 'text',
                            text: `Found ${clips.length} ${category} clip${clips.length !== 1 ? 's' : ''} for account "${currentAccountId}":\n\n${clipList}\n\n` +
                                `✨ The Statonic app is now filtered to show only ${category} clips.\n\n` +
                                `**To use in a project:** Note the clip paths and durations above. You can create a Statonic project using write_statonic_project.`,
                        }],
                };
            }
            case 'write_statonic_project': {
                const { project, filename } = args;
                // Get user's home directory
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                // Read current account from Statonic state file
                const stateFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'current-state.json');
                let currentAccountId = null;
                if (existsSync(stateFile)) {
                    try {
                        const state = JSON.parse(readFileSync(stateFile, 'utf-8'));
                        currentAccountId = state.currentAccountId;
                    }
                    catch {
                        // Ignore parse errors
                    }
                }
                if (!currentAccountId) {
                    throw new Error('No account is currently active in Statonic. Please select an account in the app first.');
                }
                // Add accountId to project if not already present
                if (!project.accountId) {
                    project.accountId = currentAccountId;
                }
                // Save to account-specific projects directory
                const projectsDir = join(homeDir, 'Library', 'Application Support', 'Statonic', 'projects', 'accounts', currentAccountId);
                if (!existsSync(projectsDir)) {
                    mkdirSync(projectsDir, { recursive: true });
                }
                const projectPath = join(projectsDir, filename.endsWith('.json') ? filename : `${filename}.json`);
                // Write the project file
                writeFileSync(projectPath, JSON.stringify(project, null, 2));
                // Write a "load request" file for Electron to pick up
                const loadRequestFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'load-project.json');
                writeFileSync(loadRequestFile, JSON.stringify({
                    projectPath,
                    requestedAt: new Date().toISOString()
                }, null, 2));
                return {
                    content: [{
                            type: 'text',
                            text: `✅ Project saved to: ${projectPath}\n\n` +
                                `📂 Account: ${currentAccountId}\n` +
                                `📂 The Statonic app should auto-load this project now.\n\n` +
                                `Project summary:\n` +
                                `- Name: ${project.name}\n` +
                                `- Canvas: ${project.canvas.width}×${project.canvas.height}\n` +
                                `- Tracks: ${project.tracks.length}\n` +
                                `- Total segments: ${project.tracks.reduce((sum, t) => sum + t.segments.length, 0)}`,
                        }],
                };
            }
            case 'analyze_statonic_library': {
                const { clip_id, metadata, force = false } = args;
                // Determine library path - try common locations
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const libraryBasePaths = [
                    join(homeDir, 'Library', 'Application Support', 'Statonic', 'clip-library'), // macOS
                    join(homeDir, 'AppData', 'Roaming', 'Statonic', 'clip-library'), // Windows
                    join(homeDir, '.config', 'Statonic', 'clip-library'), // Linux
                ];
                let libraryBasePath = '';
                for (const path of libraryBasePaths) {
                    if (existsSync(path)) {
                        libraryBasePath = path;
                        break;
                    }
                }
                if (!libraryBasePath) {
                    throw new Error('Statonic clip library not found. Make sure the app has been run and clips have been imported.');
                }
                // If metadata provided, save it
                if (clip_id && metadata) {
                    // Find the clip in accounts
                    let metadataPath = '';
                    const accountsPath = join(libraryBasePath, 'accounts');
                    if (existsSync(accountsPath)) {
                        const accounts = readdirSync(accountsPath);
                        for (const account of accounts) {
                            const clipPath = join(accountsPath, account, 'clips', clip_id, 'metadata.json');
                            if (existsSync(clipPath)) {
                                metadataPath = clipPath;
                                break;
                            }
                        }
                    }
                    if (!metadataPath) {
                        throw new Error(`Clip ${clip_id} not found in library`);
                    }
                    const existing = JSON.parse(readFileSync(metadataPath, 'utf-8'));
                    const updated = {
                        ...existing,
                        ...metadata,
                        analyzed: true,
                        analyzedAt: new Date().toISOString(),
                    };
                    writeFileSync(metadataPath, JSON.stringify(updated, null, 2));
                    return {
                        content: [{
                                type: 'text',
                                text: `Saved analysis for clip "${existing.name}" (${clip_id})`,
                            }],
                    };
                }
                // Return unanalyzed clips - scan through all accounts
                const unanalyzed = [];
                const accountsPath = join(libraryBasePath, 'accounts');
                if (existsSync(accountsPath)) {
                    const accounts = readdirSync(accountsPath);
                    for (const account of accounts) {
                        const clipsPath = join(accountsPath, account, 'clips');
                        if (!existsSync(clipsPath))
                            continue;
                        const clipDirs = readdirSync(clipsPath);
                        for (const clipId of clipDirs) {
                            const clipDir = join(clipsPath, clipId);
                            try {
                                const stat = statSync(clipDir);
                                if (!stat.isDirectory())
                                    continue;
                                const metadataPath = join(clipDir, 'metadata.json');
                                if (existsSync(metadataPath)) {
                                    const meta = JSON.parse(readFileSync(metadataPath, 'utf-8'));
                                    if (!meta.analyzed || force) {
                                        unanalyzed.push({
                                            id: clipId,
                                            accountId: account,
                                            name: meta.name,
                                            path: meta.path,
                                            duration: meta.duration,
                                            width: meta.width,
                                            height: meta.height,
                                            analyzed: meta.analyzed || false,
                                        });
                                    }
                                }
                            }
                            catch {
                                // Skip invalid clips
                            }
                        }
                    }
                }
                if (unanalyzed.length === 0) {
                    return {
                        content: [{
                                type: 'text',
                                text: force ? 'No clips found in the library.' : 'All clips in the library have been analyzed! 🎉',
                            }],
                    };
                }
                if (clip_id) {
                    // Analyze specific clip
                    const clip = unanalyzed.find(c => c.id === clip_id);
                    if (!clip) {
                        throw new Error(`Clip ${clip_id} not found or already analyzed`);
                    }
                    // Extract frames and return for analysis
                    const { timestamps, frames } = extractKeyframes(clip.path, 4);
                    if (frames.length === 0) {
                        throw new Error('Failed to extract keyframes');
                    }
                    const content = [];
                    for (let i = 0; i < frames.length; i++) {
                        content.push({ type: 'image', data: frames[i], mimeType: 'image/jpeg' });
                        content.push({ type: 'text', text: `↑ Frame at ${timestamps[i].toFixed(1)}s` });
                    }
                    content.push({
                        type: 'text',
                        text: `\nClip: ${clip.name}\nDuration: ${clip.duration.toFixed(1)}s | Size: ${clip.width}×${clip.height}\n\n` +
                            `Please analyze and provide:\n` +
                            `1. **Name**: A 3-4 word human-readable label describing the visual content — used in the clip picker UI.\n` +
                            `   Focus on distinctive visual features: clothing color, action, setting, or expression.\n` +
                            `   Examples: "blue shirt stressed", "overhead desk notes", "gizmo quiz screen", "red hoodie reading"\n` +
                            `2. Description (2-3 sentences)\n` +
                            `3. **Category**: Classify as ONE of these clip types:\n` +
                            `   - "hook": Student sitting at desk, ENTIRELY visible in frame, looking frustrated/stressed/overwhelmed\n` +
                            `   - "gizmo": Quiz/question interface visible on screen (Gizmo app = active recall learning)\n` +
                            `   - "showcase": All other content (textbook studying, overhead views, explanations, etc.)\n` +
                            `4. Tags (5-10 keywords)\n` +
                            `5. Mood (single word)\n` +
                            `6. Subject Visible (true/false)\n` +
                            `7. Subject Position (center/left/right/etc.)\n` +
                            `8. Setting (indoor-desk/outdoor/etc.)\n\n` +
                            `After analyzing, call analyze_statonic_library with clip_id="${clip_id}" and metadata including a "name" field (the 3-4 word label) to save.`,
                    });
                    return { content };
                }
                else {
                    // List all clips
                    const summary = unanalyzed.map(c => `• ${c.name} (${c.id}) [${c.accountId}] - ${c.duration.toFixed(1)}s, ${c.width}×${c.height}${c.analyzed ? ' ✓' : ''}`).join('\n');
                    const clipType = force ? 'clip' : 'unanalyzed clip';
                    return {
                        content: [{
                                type: 'text',
                                text: `Found ${unanalyzed.length} ${clipType}${unanalyzed.length !== 1 ? 's' : ''}:\n\n${summary}\n\n` +
                                    `To analyze, call analyze_statonic_library with a specific clip_id${force ? ' and force=true' : ''}.`,
                            }],
                    };
                }
            }
            case 'get_suitable_audio': {
                const { hook_duration_sec, total_duration_sec, prefer_closest = false } = args;
                // Determine audio library path
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const audioLibraryPaths = [
                    join(homeDir, 'Library', 'Application Support', 'Statonic', 'audio-library'), // macOS
                    join(homeDir, 'AppData', 'Roaming', 'Statonic', 'audio-library'), // Windows
                    join(homeDir, '.config', 'Statonic', 'audio-library'), // Linux
                ];
                let audioLibraryPath = '';
                for (const path of audioLibraryPaths) {
                    if (existsSync(path)) {
                        audioLibraryPath = path;
                        break;
                    }
                }
                if (!audioLibraryPath) {
                    throw new Error('Statonic audio library not found. Make sure the app has been run and audios have been imported.');
                }
                // Collect all audios
                const audios = [];
                const audioDirs = readdirSync(audioLibraryPath);
                for (const audioId of audioDirs) {
                    const audioDir = join(audioLibraryPath, audioId);
                    try {
                        const stat = statSync(audioDir);
                        if (!stat.isDirectory())
                            continue;
                        const metadataPath = join(audioDir, 'metadata.json');
                        if (existsSync(metadataPath)) {
                            const meta = JSON.parse(readFileSync(metadataPath, 'utf-8'));
                            audios.push({
                                id: meta.id,
                                name: meta.name,
                                path: meta.path,
                                duration: meta.duration,
                                dropTimeMs: meta.dropTimeMs || null,
                            });
                        }
                    }
                    catch {
                        // Skip invalid directories
                    }
                }
                // Filter suitable audios
                const suitableAudios = audios.filter(a => {
                    // Must have a drop time
                    if (a.dropTimeMs === null || a.dropTimeMs === undefined)
                        return false;
                    // Drop should be within ±0.1s of transition (can be slightly before or after)
                    const dropTimeSec = a.dropTimeMs / 1000;
                    if (dropTimeSec < hook_duration_sec - 0.1)
                        return false;
                    // Audio must be long enough to cover entire video
                    if (a.duration < total_duration_sec)
                        return false;
                    return true;
                });
                if (suitableAudios.length === 0) {
                    return {
                        content: [{
                                type: 'text',
                                text: `❌ No suitable audio found.\n\n` +
                                    `Requirements:\n` +
                                    `- Drop time > ${hook_duration_sec}s (transition point)\n` +
                                    `- Duration >= ${total_duration_sec}s (total video length)\n\n` +
                                    `Found ${audios.length} total audio${audios.length !== 1 ? 's' : ''} in library, but none matched the criteria.\n\n` +
                                    `Available audios:\n` +
                                    audios.map(a => `  • ${a.name}: drop at ${a.dropTimeMs ? (a.dropTimeMs / 1000).toFixed(2) + 's' : 'N/A'}, duration ${a.duration.toFixed(2)}s`).join('\n'),
                            }],
                    };
                }
                // Select audio
                let selectedAudio;
                if (prefer_closest) {
                    // Pick audio with drop time closest to hook duration
                    suitableAudios.sort((a, b) => {
                        const diffA = Math.abs(a.dropTimeMs / 1000 - hook_duration_sec);
                        const diffB = Math.abs(b.dropTimeMs / 1000 - hook_duration_sec);
                        return diffA - diffB;
                    });
                    selectedAudio = suitableAudios[0];
                }
                else {
                    // Pick randomly
                    selectedAudio = suitableAudios[Math.floor(Math.random() * suitableAudios.length)];
                }
                // Calculate audio positioning
                const dropTimeSec = selectedAudio.dropTimeMs / 1000;
                const audioStartSec = hook_duration_sec - dropTimeSec;
                const audioStartUs = Math.round(audioStartSec * 1e6);
                // Calculate source start (if audio starts before video, we need to trim from source)
                const sourceStartUs = audioStartUs < 0 ? Math.round(Math.abs(audioStartUs)) : 0;
                return {
                    content: [{
                            type: 'text',
                            text: `✅ Found suitable audio: **${selectedAudio.name}**\n\n` +
                                `Audio details:\n` +
                                `- Drop time: ${dropTimeSec.toFixed(2)}s\n` +
                                `- Duration: ${selectedAudio.duration.toFixed(2)}s\n` +
                                `- Path: ${selectedAudio.path}\n\n` +
                                `Positioning for your project:\n` +
                                `- Hook ends at: ${hook_duration_sec}s (transition point)\n` +
                                `- Audio starts at: ${audioStartSec.toFixed(3)}s ${audioStartSec < 0 ? '(starts ' + Math.abs(audioStartSec).toFixed(3) + 's before video)' : ''}\n` +
                                `- Drop hits exactly at: ${hook_duration_sec}s ✨\n\n` +
                                `**Audio segment for Statonic project:**\n` +
                                `\`\`\`json\n` +
                                `{\n` +
                                `  "id": "audio-1",\n` +
                                `  "type": "audio",\n` +
                                `  "src": "${selectedAudio.path}",\n` +
                                `  "name": "${selectedAudio.name}",\n` +
                                `  "startUs": ${audioStartUs},\n` +
                                `  "durationUs": ${Math.round(total_duration_sec * 1e6)},\n` +
                                `  "sourceStartUs": ${sourceStartUs},\n` +
                                `  "sourceDurationUs": ${Math.round(total_duration_sec * 1e6)},\n` +
                                `  "fileDurationUs": ${Math.round(selectedAudio.duration * 1e6)},\n` +
                                `  "volume": 1.0,\n` +
                                `  "dropTimeUs": ${selectedAudio.dropTimeMs * 1000}\n` +
                                `}\n` +
                                `\`\`\`\n\n` +
                                `To include in project, add an audio track:\n` +
                                `\`\`\`json\n` +
                                `{\n` +
                                `  "id": "track-audio",\n` +
                                `  "type": "audio",\n` +
                                `  "label": "AUDIO",\n` +
                                `  "segments": [/* audio segment above */]\n` +
                                `}\n` +
                                `\`\`\`\n\n` +
                                `(${suitableAudios.length} suitable audio${suitableAudios.length !== 1 ? 's' : ''} available)`,
                        }],
                };
            }
            case 'get_reference_frames': {
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const requestFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'reference-request.json');
                if (!existsSync(requestFile)) {
                    return { content: [{ type: 'text', text: 'No reference request found. Click "Copy Reference" in the editor and select a video first.' }] };
                }
                const request = JSON.parse(readFileSync(requestFile, 'utf-8'));
                const { frames, totalDuration } = request;
                const content = [
                    {
                        type: 'text',
                        text: `Analyze this ${totalDuration.toFixed(1)}s TikTok video. ${frames.length} frames follow (one every 0.5s).\n\n` +
                            `STEP 1 — For EVERY frame, read ALL text on screen AND estimate each text block's vertical centre as a fraction from the top of the frame (0.0=top, 0.5=centre, 1.0=bottom):\n` +
                            `  t=0.0s: text="how to study\\nBIOLOGY\\nand get 99/100\\nwhen you hate it" pos≈0.55\n` +
                            `  t=5.0s: text="Students who follow\\nme and use:" pos≈0.52  text="GAMIFICATION" pos≈0.65\n` +
                            `(EXAMPLE only — fill in what you ACTUALLY see. Hook frames have text too — do not skip them.)\n\n` +
                            `STEP 2 — Group consecutive frames with IDENTICAL text into one slot:\n` +
                            `  • Only TEXT CHANGES create new slots — background clip cuts do NOT\n` +
                            `  • detectedText = the COMPLETE text you wrote in Step 1, ALL lines, \\n between lines\n` +
                            `  • NEVER leave detectedText empty if you saw text in those frames\n` +
                            `  • clipType: first section = "hook", then "gizmo" (app/quiz on screen) or "showcase" (person studying)\n\n` +
                            `STEP 3 — Convert your position measurements to y coordinates (formula: y = 1 - 2 * pos):\n` +
                            `  • hookTextY: y for the hook section text (e.g. pos=0.55 → y=-0.1)\n` +
                            `  • spanningTextY: y for the persistent prefix text (e.g. pos=0.52 → y=-0.04) — MUST be greater than slotTextY\n` +
                            `  • slotTextY: y for the per-slot technique name (e.g. pos=0.65 → y=-0.3) — MUST be less than spanningTextY\n\n` +
                            `STEP 4 — Call write_reference_result with slots, spanning_texts, hookTextY, spanningTextY, slotTextY. thumbnailPath: leave as ""\n\n` +
                            `Frames:`,
                    },
                ];
                for (let i = 0; i < frames.length; i++) {
                    const f = frames[i];
                    content.push({ type: 'text', text: `t=${f.timeSec.toFixed(1)}s` });
                    if (existsSync(f.framePath)) {
                        const b64 = readFileSync(f.framePath).toString('base64');
                        content.push({ type: 'image', data: b64, mimeType: 'image/jpeg' });
                    }
                    else {
                        content.push({ type: 'text', text: '(frame missing)' });
                    }
                }
                return { content };
            }
            case 'write_reference_result': {
                let { slots, spanning_texts: rawSpanning, hookTextY, spanningTextY, slotTextY } = args;
                let spanning_texts = rawSpanning ?? [];
                // ── Merge all pre-technique slots into one hook slot ──────────────────
                // "Pre-technique" = all slots before the first slot that has a non-empty detectedText
                const firstTechIdx = slots.findIndex((s) => s.detectedText && s.detectedText.trim());
                if (firstTechIdx > 1) {
                    const preTech = slots.slice(0, firstTechIdx);
                    const techSlots = slots.slice(firstTechIdx);
                    const hookEnd = techSlots[0].startSec;
                    const hookSpan = spanning_texts.find((st) => st.fromSlot === 0);
                    const hookText = hookSpan?.text ?? preTech.find((s) => s.detectedText?.trim())?.detectedText ?? '';
                    slots = [
                        { ...preTech[0], startSec: 0, durationSec: hookEnd, clipType: 'hook', detectedText: hookText },
                        ...techSlots,
                    ];
                    spanning_texts = spanning_texts.filter((st) => st.fromSlot !== 0);
                }
                else if (firstTechIdx === -1 && slots.length > 0) {
                    slots[0] = { ...slots[0], clipType: 'hook' };
                }
                if (slots.length > 0)
                    slots[0] = { ...slots[0], clipType: 'hook' };
                // ── Move hook spanning_text into slot 0's detectedText ────────────────
                // (handles case where Claude outputs 1 hook slot + spanning_text for it)
                const hookSpan = spanning_texts.find((st) => st.fromSlot === 0);
                if (hookSpan && !slots[0]?.detectedText?.trim()) {
                    slots[0] = { ...slots[0], detectedText: hookSpan.text };
                    spanning_texts = spanning_texts.filter((st) => st !== hookSpan);
                }
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const resultFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'reference-result.json');
                writeFileSync(resultFile, JSON.stringify({
                    resolvedAt: new Date().toISOString(), slots, spanning_texts,
                    ...(hookTextY !== undefined && { hookTextY }),
                    ...(spanningTextY !== undefined && { spanningTextY }),
                    ...(slotTextY !== undefined && { slotTextY }),
                }, null, 2));
                return {
                    content: [{
                            type: 'text',
                            text: `✅ Reference analysis written (${slots.length} slot${slots.length !== 1 ? 's' : ''}).\n` +
                                `The editor modal will now show the detected structure for you to assign clips.`,
                        }],
                };
            }
            case 'list_templates': {
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const templatesDir = join(homeDir, 'Library', 'Application Support', 'Statonic', 'templates');
                if (!existsSync(templatesDir)) {
                    return { content: [{ type: 'text', text: 'No templates directory found. Create templates at: ' + templatesDir }] };
                }
                const files = readdirSync(templatesDir).filter(f => f.endsWith('.json'));
                const templates = files.map(f => {
                    try {
                        const t = JSON.parse(readFileSync(join(templatesDir, f), 'utf-8'));
                        return { id: t.id, name: t.name, description: t.description, slot_count: t.slots?.length ?? 0, total_duration_sec: t.total_duration_sec };
                    }
                    catch {
                        return { id: f.replace('.json', ''), name: f, description: 'Parse error', slot_count: 0, total_duration_sec: 0 };
                    }
                });
                return {
                    content: [{
                            type: 'text',
                            text: `Found ${templates.length} template(s):\n\n` +
                                templates.map(t => `**${t.id}** — ${t.name}\n  ${t.description}\n  ${t.slot_count} slots, ${t.total_duration_sec}s total`).join('\n\n'),
                        }],
                };
            }
            case 'use_template': {
                const { template_id, slots: slotOverrides = [], project_name } = args;
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const templatesDir = join(homeDir, 'Library', 'Application Support', 'Statonic', 'templates');
                const templatePath = join(templatesDir, `${template_id}.json`);
                if (!existsSync(templatePath)) {
                    throw new Error(`Template "${template_id}" not found at ${templatePath}`);
                }
                const template = JSON.parse(readFileSync(templatePath, 'utf-8'));
                // Load clip library to auto-fill empty slots
                const stateFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'current-state.json');
                let currentAccountId = null;
                if (existsSync(stateFile)) {
                    try {
                        currentAccountId = JSON.parse(readFileSync(stateFile, 'utf-8')).currentAccountId;
                    }
                    catch { }
                }
                if (!currentAccountId)
                    throw new Error('No account active in Statonic. Select an account first.');
                const clipLibraryDir = join(homeDir, 'Library', 'Application Support', 'Statonic', 'clip-library', 'accounts', currentAccountId, 'clips');
                const clipsByCategory = {};
                if (existsSync(clipLibraryDir)) {
                    const clipIds = readdirSync(clipLibraryDir);
                    for (const clipId of clipIds) {
                        const metaPath = join(clipLibraryDir, clipId, 'metadata.json');
                        const srcDir = join(clipLibraryDir, clipId);
                        if (!existsSync(metaPath))
                            continue;
                        try {
                            const meta = JSON.parse(readFileSync(metaPath, 'utf-8'));
                            const category = meta.category || 'unknown';
                            const files = readdirSync(srcDir).filter(f => /\.(mp4|mov|m4v)$/i.test(f));
                            if (files.length === 0)
                                continue;
                            const clipPath = join(srcDir, files[0]);
                            if (!clipsByCategory[category])
                                clipsByCategory[category] = [];
                            clipsByCategory[category].push({
                                id: clipId,
                                path: clipPath,
                                name: meta.name || files[0],
                                durationUs: Math.round((meta.duration || 5) * 1e6),
                                width: meta.width || 1080,
                                height: meta.height || 1920,
                            });
                        }
                        catch { }
                    }
                }
                function uid() { return randomBytes(4).toString('hex'); }
                function pickClip(category) {
                    const pool = clipsByCategory[category] || [];
                    if (pool.length === 0)
                        return null;
                    return pool[Math.floor(Math.random() * pool.length)];
                }
                // Build project tracks
                const videoTrack = { id: uid(), type: 'video', label: 'VIDEO', segments: [] };
                const textTrack = { id: uid(), type: 'text', label: 'TEXT', segments: [] };
                for (const slot of template.slots) {
                    const override = slotOverrides.find((o) => o.slot_id === slot.slot_id);
                    const startUs = Math.round(slot.start_sec * 1e6);
                    const durationUs = Math.round(slot.duration_sec * 1e6);
                    // Resolve clip
                    let clipPath = null;
                    let clipName = 'clip';
                    let clipWidth = 1080;
                    let clipHeight = 1920;
                    let clipFileDurationUs = durationUs;
                    if (override?.clip_id) {
                        // Try to find clip by id in library
                        const clipDir = existsSync(clipLibraryDir) ? join(clipLibraryDir, override.clip_id) : '';
                        if (clipDir && existsSync(clipDir)) {
                            const files = readdirSync(clipDir).filter(f => /\.(mp4|mov|m4v)$/i.test(f));
                            if (files.length > 0) {
                                const metaPath = join(clipDir, 'metadata.json');
                                const meta = existsSync(metaPath) ? JSON.parse(readFileSync(metaPath, 'utf-8')) : {};
                                clipPath = join(clipDir, files[0]);
                                clipName = meta.name || files[0];
                                clipWidth = meta.width || 1080;
                                clipHeight = meta.height || 1920;
                                clipFileDurationUs = Math.round((meta.duration || 5) * 1e6);
                            }
                        }
                    }
                    if (!clipPath) {
                        const auto = pickClip(slot.clip_category);
                        if (auto) {
                            clipPath = auto.path;
                            clipName = auto.name;
                            clipWidth = auto.width;
                            clipHeight = auto.height;
                            clipFileDurationUs = auto.durationUs;
                        }
                    }
                    if (clipPath) {
                        videoTrack.segments.push({
                            id: uid(), type: 'video',
                            src: clipPath, name: clipName,
                            startUs, durationUs,
                            sourceStartUs: 0, sourceDurationUs: durationUs, fileDurationUs: clipFileDurationUs,
                            sourceWidth: clipWidth, sourceHeight: clipHeight,
                            clipX: 0, clipY: 0, clipScale: 1,
                            cropLeft: 0, cropRight: 0, cropTop: 0, cropBottom: 0,
                        });
                    }
                    // Text overlay
                    const textContent = override?.text ?? slot.text?.example ?? '';
                    if (textContent) {
                        textTrack.segments.push({
                            id: uid(), type: 'text', text: textContent,
                            startUs, durationUs,
                            x: 0, y: slot.text?.y ?? 0.28,
                            fontSize: slot.text?.fontSize ?? 85,
                            color: '#ffffff', bold: false, italic: false,
                            strokeEnabled: false, strokeColor: '#000000',
                            textAlign: 'center', textScale: 1,
                        });
                    }
                }
                const finalName = project_name || `${template.name} - ${new Date().toLocaleDateString()}`;
                const project = {
                    name: finalName,
                    accountId: currentAccountId,
                    canvas: { width: 1080, height: 1920 },
                    tracks: [videoTrack, textTrack],
                };
                const projectsDir = join(homeDir, 'Library', 'Application Support', 'Statonic', 'projects', 'accounts', currentAccountId);
                mkdirSync(projectsDir, { recursive: true });
                const safeFilename = finalName.replace(/[^a-zA-Z0-9-_ ]/g, '').replace(/\s+/g, '-').toLowerCase();
                const projectPath = join(projectsDir, `${safeFilename}.json`);
                writeFileSync(projectPath, JSON.stringify(project, null, 2));
                const loadRequestFile = join(homeDir, 'Library', 'Application Support', 'Statonic', 'load-project.json');
                writeFileSync(loadRequestFile, JSON.stringify({ projectPath, requestedAt: new Date().toISOString() }, null, 2));
                const filledSlots = template.slots.map((s) => {
                    const ov = slotOverrides.find((o) => o.slot_id === s.slot_id);
                    return `  ${s.slot_id}: ${ov?.text ?? s.text?.example ?? '(no text)'} | ${ov?.clip_id ? 'clip: ' + ov.clip_id : 'auto-selected ' + s.clip_category}`;
                }).join('\n');
                return {
                    content: [{
                            type: 'text',
                            text: `✅ Project created from template "${template_id}":\n\n` +
                                `Name: ${finalName}\n` +
                                `Path: ${projectPath}\n` +
                                `Slots:\n${filledSlots}\n\n` +
                                `The Statonic app should auto-load this project now.\n` +
                                `Use render_preview to verify text positioning.`,
                        }],
                };
            }
            case 'generate_hook_options': {
                const { topic, count = 5 } = args;
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const knowledgePath = join(homeDir, 'Library', 'Application Support', 'Statonic', 'hook-knowledge.json');
                let knowledge = { formulas: [], learned_examples: [] };
                if (existsSync(knowledgePath)) {
                    try {
                        knowledge = JSON.parse(readFileSync(knowledgePath, 'utf-8'));
                    }
                    catch { }
                }
                const anthropic = new Anthropic();
                const formulaList = knowledge.formulas.map(f => `- ${f.id}: "${f.pattern}" (e.g. "${f.example}")${f.notes ? ' — ' + f.notes : ''}`).join('\n');
                const learnedList = knowledge.learned_examples.slice(0, 5).map((e) => `- "${e.extracted_text}" (formula: ${e.formula}, topic: ${e.topic})`).join('\n');
                const prompt = `You are a TikTok/Reels hook writer. Generate ${count} hook text options for the topic: "${topic}".

Available proven formulas:
${formulaList}
${learnedList ? '\nLearned from trending videos:\n' + learnedList : ''}

Rules:
- 2-3 lines of text, 4-7 words per line
- Use CAPS for ONE key emphasis word per line max
- Break lines with \\n
- Output JSON array: [{"text": "line1\\nline2\\nline3", "formula": "formula_id", "explanation": "why this works"}]
- Only output the JSON array, no other text`;
                const response = await anthropic.messages.create({
                    model: 'claude-opus-4-6',
                    max_tokens: 1024,
                    messages: [{ role: 'user', content: prompt }],
                });
                const rawText = response.content[0].type === 'text' ? response.content[0].text.trim() : '[]';
                let options = [];
                try {
                    const jsonMatch = rawText.match(/\[[\s\S]*\]/);
                    options = jsonMatch ? JSON.parse(jsonMatch[0]) : [];
                }
                catch {
                    options = [];
                }
                const formatted = options.map((o, i) => `${i + 1}. [${o.formula}]\n   Text: "${o.text}"\n   Why: ${o.explanation}`).join('\n\n');
                return {
                    content: [{
                            type: 'text',
                            text: `Hook options for topic "${topic}":\n\n${formatted}\n\n` +
                                `To use: copy the "text" value into write_statonic_project as the hook text overlay.`,
                        }],
                };
            }
            case 'learn_from_hook_video': {
                const { video_path } = args;
                if (!existsSync(video_path))
                    throw new Error(`Video not found: ${video_path}`);
                // Extract 3 frames from first 6 seconds
                const frames = [];
                const timestamps = [1, 2.5, 4.5];
                for (const t of timestamps) {
                    try {
                        frames.push(extractFrame(video_path, t));
                    }
                    catch { }
                }
                if (frames.length === 0)
                    throw new Error('Could not extract frames from video');
                const homeDir = process.env.HOME || process.env.USERPROFILE || '~';
                const knowledgePath = join(homeDir, 'Library', 'Application Support', 'Statonic', 'hook-knowledge.json');
                let knowledge = { formulas: [], learned_examples: [] };
                if (existsSync(knowledgePath)) {
                    try {
                        knowledge = JSON.parse(readFileSync(knowledgePath, 'utf-8'));
                    }
                    catch { }
                }
                const knownFormulas = knowledge.formulas.map(f => f.id).join(', ');
                const anthropic = new Anthropic();
                const content = [
                    { type: 'text', text: `Analyze these frames from the first ~5 seconds of a TikTok/Reels video. Known hook formulas: ${knownFormulas}.\n\nAnswer in JSON: {"extracted_text": "exact text visible on screen", "formula": "closest_formula_id or new_formula_name", "topic": "subject matter", "effectiveness_notes": "why this hook works"}` },
                    ...frames.map(b64 => ({ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: b64 } })),
                ];
                const response = await anthropic.messages.create({
                    model: 'claude-opus-4-6',
                    max_tokens: 512,
                    messages: [{ role: 'user', content }],
                });
                const rawText = response.content[0].type === 'text' ? response.content[0].text.trim() : '{}';
                let analysis = {};
                try {
                    const jsonMatch = rawText.match(/\{[\s\S]*\}/);
                    analysis = jsonMatch ? JSON.parse(jsonMatch[0]) : {};
                }
                catch { }
                const learned = {
                    formula: analysis.formula || 'unknown',
                    extracted_text: analysis.extracted_text || '',
                    topic: analysis.topic || '',
                    effectiveness_notes: analysis.effectiveness_notes || '',
                    source: video_path,
                    learned_at: new Date().toISOString(),
                };
                knowledge.learned_examples = knowledge.learned_examples || [];
                knowledge.learned_examples.push(learned);
                writeFileSync(knowledgePath, JSON.stringify(knowledge, null, 2));
                return {
                    content: [{
                            type: 'text',
                            text: `✅ Learned from hook video:\n\n` +
                                `Text detected: "${learned.extracted_text}"\n` +
                                `Formula: ${learned.formula}\n` +
                                `Topic: ${learned.topic}\n` +
                                `Notes: ${learned.effectiveness_notes}\n\n` +
                                `Appended to hook-knowledge.json (${knowledge.learned_examples.length} learned examples total).`,
                        }],
                };
            }
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
    catch (e) {
        return {
            content: [{ type: 'text', text: `Error: ${e.message}` }],
            isError: true,
        };
    }
});
const transport = new StdioServerTransport();
await server.connect(transport);
