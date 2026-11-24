mod ollama;

use glob::glob;
use ollama::{Message, Ollama};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{{Component, Path, PathBuf}};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn system_prompt() -> String {
    let cwd = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let environment = if cfg!(windows) {
        "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix Bash or PowerShell syntax"
    } else {
        "Unix-like: the bash tool runs the system shell"
    };
    format!(
        "You are a coding agent at {cwd}. Environment: {environment}. Use bash to solve tasks. Act, don't explain."
    )
}


fn read_pipe(pipe: Option<impl Read>) -> String {
    let mut bytes = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = pipe.read_to_end(&mut bytes);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn run_bash(command: &str) -> String {
    const DANGEROUS: [&str; 5] = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"];
    if DANGEROUS.iter().any(|item| command.contains(item)) {
        return "Error: Dangerous command blocked".to_string();
    }

    let mut cmd = if cfg!(windows) {
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", command]);
        cmd
    } else {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", command]);
        cmd
    };
    if let Ok(cwd) = std::env::current_dir() {
        cmd.current_dir(cwd);
    }

    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(err) => return format!("Error: {err}"),
    };
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let stdout_task = thread::spawn(move || read_pipe(stdout_pipe));
    let stderr_task = thread::spawn(move || read_pipe(stderr_pipe));

    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() >= Duration::from_secs(120) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_task.join();
                let _ = stderr_task.join();
                return "Error: Timeout (120s)".to_string();
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_task.join();
                let _ = stderr_task.join();
                return format!("Error: {err}");
            }
        }
    }

    let stdout = stdout_task.join().unwrap_or_default();
    let stderr = stderr_task.join().unwrap_or_default();
    let combined = format!("{stdout}{stderr}");
    let trimmed = combined.trim();
    if trimmed.is_empty() {
        "(no output)".to_string()
    } else {
        trimmed.chars().take(50_000).collect()
    }
}

//

/// 安全路徑解析（支援尚未建立的新檔案，防止 ../ 路徑穿越）
fn safe_path(p: &str) -> Result<PathBuf, String> {
    let workdir = env::current_dir()
        .and_then(|d| d.canonicalize())
        .map_err(|e| format!("Failed to get workdir: {}", e))?;

    let target = workdir.join(p);

    // 1. 往上尋找第一個「已經存在」的祖先目錄並解析真實路徑
    let mut existing = target.as_path();
    let mut tail = Vec::new();
    while !existing.exists() {
        if let Some(name) = existing.file_name() {
            tail.push(name);
            existing = existing.parent().unwrap_or(Path::new(""));
        } else {
            break;
        }
    }

    let mut resolved = existing
        .canonicalize()
        .map_err(|e| format!("Invalid path '{}': {}", p, e))?;

    // 2. 將尚未建立的子路徑接回去，並檢查是否包含惡意的 `..`
    for part in tail.into_iter().rev() {
        let comp = Path::new(part).components().next();
        if matches!(comp, Some(Component::ParentDir)) {
            return Err(format!("Path escapes workspace: {}", p));
        }
        resolved.push(part);
    }

    // 3. 確保最終路徑依然在 workdir 內部
    if !resolved.starts_with(&workdir) {
        return Err(format!("Path escapes workspace: {}", p));
    }

    Ok(resolved)
}

pub fn run_read(path: &str, limit: Option<usize>) -> String {
    let file_path = match safe_path(path) {
        Ok(p) => p,
        Err(e) => return format!("Error: {}", e),
    };

    let content = match fs::read_to_string(&file_path) {
        Ok(c) => c,
        Err(e) => return format!("Error: {}", e),
    };

    let lines: Vec<&str> = content.lines().collect();
    if let Some(lim) = limit {
        if lim < lines.len() {
            let mut result: Vec<String> = lines[..lim].iter().map(|s| s.to_string()).collect();
            result.push(format!("... ({} more lines)", lines.len() - lim));
            return result.join("\n");
        }
    }

    lines.join("\n")
}

pub fn run_write(path: &str, content: &str) -> String {
    let file_path = match safe_path(path) {
        Ok(p) => p,
        Err(e) => return format!("Error: {}", e),
    };

    if let Some(parent) = file_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return format!("Error: {}", e);
        }
    }

    match fs::write(&file_path, content) {
        Ok(_) => format!("Wrote {} bytes to {}", content.len(), path),
        Err(e) => format!("Error: {}", e),
    }
}

pub fn run_edit(path: &str, old_text: &str, new_text: &str) -> String {
    let file_path = match safe_path(path) {
        Ok(p) => p,
        Err(e) => return format!("Error: {}", e),
    };

    let text = match fs::read_to_string(&file_path) {
        Ok(t) => t,
        Err(e) => return format!("Error: {}", e),
    };

    if !text.contains(old_text) {
        return format!("Error: text not found in {}", path);
    }

    // replacen(..., 1) 對應 Python 的 text.replace(old_text, new_text, 1)
    let updated = text.replacen(old_text, new_text, 1);

    match fs::write(&file_path, updated) {
        Ok(_) => format!("Edited {}", path),
        Err(e) => format!("Error: {}", e),
    }
}


pub fn run_glob(pattern: &str) -> String {
    // 1. 直接從環境取得當前工作目錄，並解析真實路徑
    let canonical_workdir = match env::current_dir().and_then(|dir| dir.canonicalize()) {
        Ok(p) => p,
        Err(e) => return format!("Error: {}", e),
    };

    // 2. 組合搜尋路徑
    let full_pattern = canonical_workdir.join(pattern);
    let pattern_str = match full_pattern.to_str() {
        Some(s) => s,
        None => return "Error: Invalid UTF-8 in path".to_string(),
    };

    // 3. 執行 glob 搜尋
    let entries = match glob(pattern_str) {
        Ok(paths) => paths,
        Err(e) => return format!("Error: {}", e),
    };

    let mut matches = BTreeSet::new();

    for entry in entries {
        match entry {
            Ok(path) => {
                if let Ok(resolved) = path.canonicalize() {
                    // 確保檔案在工作目錄內，並轉回相對路徑
                    if let Ok(rel_path) = resolved.strip_prefix(&canonical_workdir) {
                        matches.insert(rel_path.to_string_lossy().into_owned());
                    }
                }
            }
            Err(e) => return format!("Error: {}", e),
        }
    }

    // 4. 處理空結果
    if matches.is_empty() {
        return "(no matches)".to_string();
    }

    // 5. 限制最多顯示 200 筆
    let total = matches.len();
    let mut shown: Vec<String> = matches.into_iter().take(200).collect();

    if total > 200 {
        shown.push("... (more matches omitted; narrow the pattern)".to_string());
    }

    shown.join("\n")
}

fn tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command. Do NOT use bash to read, write, edit, or search files when dedicated tools are available.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" }
                    },
                    "required": ["command"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read file contents. Prefer this over cat/head/tail in bash.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "limit": { "type": "integer" }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file (creates parent directories if needed).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace exact text in a file once.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "old_text": { "type": "string" },
                        "new_text": { "type": "string" }
                    },
                    "required": ["path", "old_text", "new_text"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "glob",
                "description": "Fast file path pattern matching tool; ** matches recursively (e.g., **/*.rs). ALWAYS use this instead of find or ls in bash.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string" }
                    },
                    "required": ["pattern"]
                }
            }
        }),
    ]
}


type ToolHandler = fn(&Value) -> String;

fn get_tool_handler(name: &str) -> Option<ToolHandler> {
    match name {
        "bash" => Some(handle_bash),
        "read_file" => Some(handle_read),
        "write_file" => Some(handle_write),
        "edit_file" => Some(handle_edit),
        "glob" => Some(handle_glob),
        _ => None,
    }
}

fn handle_bash(args: &Value) -> String {
    let Some(command) = args.get("command").and_then(Value::as_str) else {
        return "Error: missing 'command' argument".to_string();
    };
    println!("\x1b[33m$ {command}\x1b[0m");
    run_bash(command)
}

fn handle_read(args: &Value) -> String {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return "Error: missing 'path' argument".to_string();
    };
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    println!("\x1b[33m[read] {path}\x1b[0m");
    run_read(path, limit)
}

fn handle_write(args: &Value) -> String {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return "Error: missing 'path' argument".to_string();
    };
    let Some(content) = args.get("content").and_then(Value::as_str) else {
        return "Error: missing 'content' argument".to_string();
    };
    println!("\x1b[33m[write] {path}\x1b[0m");
    run_write(path, content)
}

fn handle_edit(args: &Value) -> String {
    let Some(path) = args.get("path").and_then(Value::as_str) else {
        return "Error: missing 'path' argument".to_string();
    };
    let Some(old_text) = args.get("old_text").and_then(Value::as_str) else {
        return "Error: missing 'old_text' argument".to_string();
    };
    let Some(new_text) = args.get("new_text").and_then(Value::as_str) else {
        return "Error: missing 'new_text' argument".to_string();
    };
    println!("\x1b[33m[edit] {path}\x1b[0m");
    run_edit(path, old_text, new_text)
}

fn handle_glob(args: &Value) -> String {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return "Error: missing 'pattern' argument".to_string();
    };
    println!("\x1b[33m[glob] {pattern}\x1b[0m");
    run_glob(pattern)
}

async fn agent_loop(
    client: &Ollama,
    messages: &mut Vec<Message>
) -> Result<(), Box<dyn std::error::Error>> {
    let system = system_prompt();
    let tools = tools();

    loop {
        let response = client
            .messages()
            .create(messages.clone())
            .system(system.clone())
            .tools(tools.clone())
            .max_tokens(8000)
            .send()
            .await?;

        let assistant = response.message;
        let tool_calls = assistant.tool_calls.clone().unwrap_or_default();
        messages.push(assistant);

        if tool_calls.is_empty() {
            return Ok(());
        }

        for call in tool_calls {
            let name = call.function.name.clone();
            let output = if let Some(handler) = get_tool_handler(&name) {
                let output = handler(&call.function.arguments);
                let preview: String = output.chars().take(200).collect();
                println!("{preview}");
                output
            } else {
                format!("Error: unknown tool {name}")
            };

            messages.push(Message {
                role: "tool".to_string(),
                content: output,
                tool_name: Some(name),
                ..Default::default()
            });
        }    
    }
}


fn print_final_text(messages: &[Message]) {
    if let Some(last) = messages.last() {
        if last.role == "assistant" && !last.content.is_empty() {
            println!("{}", last.content);
        }
    }
    println!();
}

#[tokio::main]
async fn main() {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("MODEL_ID").unwrap_or_else(|_| "gemma4".to_string());
    let client = Ollama::new(host, model);

    println!("s02: Tool use - four tools added to s01");
    println!("Enter a question, press Enter to send. Type q to quit.");

    let mut history = Vec::new();
    loop {
        print!("s02 >> ");
        io::stdout().flush().expect("Can't flush stdout");

        let mut input = String::new();
        match std::io::stdin().read_line(&mut input) {
            Ok(_) => {
                let query = input.trim();
                if query.is_empty() {
                    continue;
                }
                if matches!(query.to_lowercase().as_str(), "q" | "exit") {
                    break;
                }

                history.push(Message {
                    role: "user".to_string(),
                    content: query.to_string(),
                    ..Default::default()
                });

                if let Err(err) = agent_loop(&client, &mut history).await {
                    eprintln!("Error: {err}");
                    println!();
                    continue;
                }

                print_final_text(&history);

            }
            Err(err) => {
                eprintln!("Input error: {err}");
                break;
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::run_bash;

    #[test]
    fn blocks_dangerous_commands_without_running_them() {
        for command in ["sudo ls", "shutdown -h now", "reboot", "echo hi > /dev/null", "rm -rf /"] {
            assert_eq!(run_bash(command), "Error: Dangerous command blocked", "{command}");
        }
    }

    #[test]
    fn runs_a_shell_command_and_reports_empty_output() {
        assert_eq!(run_bash("echo hello"), "hello");
        assert_eq!(run_bash("true"), "(no output)");
    }
}

