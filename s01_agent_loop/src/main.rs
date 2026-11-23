//! s01_agent_loop — the agent loop.
//!
//!     while True:
//!         response = LLM(messages, tools)
//!         if response contains no tool_use:
//!             break
//!         execute tools
//!         append results

mod ollama;

use ollama::{Message, Ollama};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde_json::{json, Value};
use std::io::Read;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn tools() -> Vec<Value> {
    vec![json!({
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }
        }
    })]
}

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

fn read_pipe(pipe: Option<impl Read>) -> String {
    let mut bytes = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = pipe.read_to_end(&mut bytes);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

async fn agent_loop(
    client: &Ollama,
    messages: &mut Vec<Message>,
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
            let output = if name == "bash" {
                match call
                    .function
                    .arguments
                    .get("command")
                    .and_then(Value::as_str)
                {
                    Some(command) => {
                        println!("\x1b[33m$ {command}\x1b[0m");
                        let output = run_bash(command);
                        let preview: String = output.chars().take(200).collect();
                        println!("{preview}");
                        output
                    }
                    None => "Error: missing command".to_string(),
                }
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
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("s01: Agent Loop");
    println!("Enter a question, press Enter to send. Type q to quit.\n");

    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let model = std::env::var("MODEL_ID").unwrap_or_else(|_| "gemma4".to_string());
    let client = Ollama::new(host, model);

    let mut rl = DefaultEditor::new()?;
    let mut history = Vec::new();

    loop {
        let readline = rl.readline("\x1b[36ms01 >> \x1b[0m");
        match readline {
            Ok(line) => {
                let query = line.trim();
                if query.is_empty() {
                    continue;
                }
                if matches!(query.to_lowercase().as_str(), "q" | "exit") {
                    break;
                }
                let _ = rl.add_history_entry(query);

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
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(err) => return Err(err.into()),
        }
    }

    Ok(())
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
