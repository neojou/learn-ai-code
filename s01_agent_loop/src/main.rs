mod ollama;

use ollama::{Message, Ollama};
use serde_json::{json, Value};
use std::process::Command;

// 定義 TOOLS（轉換為 Ollama / OpenAI 相容的 Function Schema）
fn get_tools() -> Vec<Value> {
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
                "required": ["command"],
            },
        }
    })]
}

// 對應 Python 的 run_bash()
fn run_bash(command: &str) -> String {
    // 依據作業系統選擇 shell（Linux/macOS 用 sh，Windows 可改為 cmd /C）
    let output = Command::new("sh").arg("-c").arg(command).output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.is_empty() {
                stdout.to_string()
            } else {
                format!("{}\nSTDERR:\n{}", stdout, stderr)
            }
        }
        Err(e) => format!("Failed to execute command: {}", e),
    }
}

// 對應 Python 的 agent_loop(messages: list)
async fn agent_loop(
    client: &Ollama,
    system: &str,
    messages: &mut Vec<Message>,
) -> Result<(), Box<dyn std::error::Error>> {
    let tools = get_tools();

    loop {
        let response = client
            .messages()
            .create(messages.clone())
            .system(system)
            .tools(tools.clone())
            .max_tokens(8000)
            .send()
            .await?;

        // 1. 將 assistant 的回覆加入對話紀錄（包含可能的 tool_calls）
        messages.push(response.message.clone());

        // 2. 檢查模型是否有呼叫工具，若無則結束迴圈
        let Some(tool_calls) = response.message.tool_calls else {
            if !response.message.content.is_empty() {
                println!("\n[Assistant]: {}", response.message.content);
            }
            return Ok(());
        };

        if tool_calls.is_empty() {
            return Ok(());
        }

        // 3. 逐一執行工具並將結果加回 messages
        for call in tool_calls {
            if call.function.name == "bash" {
                let cmd = call.function.arguments["command"]
                    .as_str()
                    .unwrap_or("");

                // 印出黃色指令提示
                println!("\x1b[33m$ {}\x1b[0m", cmd);

                let output = run_bash(cmd);

                // 安全截取前 200 個字元（避免中文 UTF-8 切片 panic）
                let preview: String = output.chars().take(200).collect();
                println!("{}", preview);

                // Ollama 規定工具執行結果需以 role: "tool" 回傳
                messages.push(Message {
                    role: "tool".to_string(),
                    content: output,
                    tool_calls: None,
                });
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 確保使用的模型支援 Tool Calling（例如 llama3.1, qwen2.5-coder, mistral 等）
    let client = Ollama::new("http://localhost:11434", "gemma4");
    let system_prompt = "You are a helpful assistant with bash access.";

    let mut messages = vec![Message {
        role: "user".to_string(),
        content: "請幫我列出目前資料夾下有哪些檔案，並告訴我現在的系統時間。".to_string(),
        tool_calls: None,
    }];

    agent_loop(&client, system_prompt, &mut messages).await?;

    Ok(())
}


/*
mod ollama;

use ollama::{Ollama, Message};

use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化 Client
    let client = Ollama::new("http://localhost:11434", "gemma4");

    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "請幫我查詢台北天氣".to_string(),
        }
    ];

    // 呼叫 client.messages().create(...)
    let response = client
        .messages()
        .create(messages)
        .system("你是一個有用的助手")
        .max_tokens(8000)
        .send()
        .await?;

    println!("模型回覆: {}", response.message.content);
    Ok(())
}


#[derive(Debug, Clone)]
struct Message {
    role: String,
    content: String,
}

fn agent_loop(client: Ollama, messages: Vec<Message>) {
    loop {
        let response = client.
    }
}

fn main() -> rustyline::Result<()> {
    println!("s01: Agent Loop");
    println!("Enter a question, press Enter to send. Type q to quit.");

    let mut rl = DefaultEditor::new()?;
    
    let mut history: Vec<Message> = Vec::new();

    loop {
        // \x1b[36m 為青色 (Cyan)，\x1b[0m 為重置顏色
        // rustyline 會自動識別 ANSI 顏色碼，不需額外包裹 \001 \002
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
                    role: String::from("user"),
                    content: query.to_string(),
                });

                println!("目前歷史紀錄筆數: {}", history.len());
                println!("最新一筆紀錄: {:?}", history.last());

                agent_loop(history)
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Read Input Error: {:?}", err);
                break;
            }
        }
    }

    println!("\n最終對話紀錄: {:#?}", history);

    Ok(())
}

*/


