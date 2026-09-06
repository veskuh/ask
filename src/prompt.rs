//! Environment detection and centralized prompt formatting for all LLM providers.

pub fn parse_linux_distro(content: &str) -> Option<String> {
    for line in content.lines() {
        if let Some(val) = line.strip_prefix("PRETTY_NAME=") {
            let trimmed = val.trim().trim_matches(|c| c == '"' || c == '\'').trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Detects the host operating system, shell, and current working directory.
pub fn get_env_context() -> (String, String, String) {
    let os_name = std::env::consts::OS;
    let mut distro = String::new();

    if os_name == "linux"
        && let Ok(content) = std::fs::read_to_string("/etc/os-release")
        && let Some(parsed) = parse_linux_distro(&content)
    {
        distro = parsed;
    }

    let context_os = if distro.is_empty() {
        os_name.to_string()
    } else {
        format!("{} ({})", os_name, distro)
    };

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    (context_os, shell, cwd)
}

/// Builds the system/user prompt for asking a CLI question.
pub fn build_command_prompt(question: &str) -> String {
    let (os, shell, cwd) = get_env_context();
    format!(
        "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n\n\
         Task: Provide the most common command line command for this query. \
         If there are other very distinct or necessary alternative ways, you may list them as well, each on a new line starting with 'Command: '. \
         Otherwise, provide just the single raw command.\n\n\
         Ensure the command is compatible with the specified OS and Shell. \
         Do not include any markdown formatting, backticks, or explanations. \
         Query: {}",
        os, shell, cwd, question
    )
}

/// Builds the prompt for refining the previously generated command.
pub fn build_refine_prompt(last_command: &str, refinement: &str) -> String {
    let (os, shell, cwd) = get_env_context();
    format!(
        "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n- Previous Command: {}\n\n\
         Task: Based on the previous command and the following refinement instruction, provide the most direct updated command line command. \
         If there are other very distinct or necessary alternative refinements, you may list them as well, each on a new line starting with 'Command: '. \
         Otherwise, provide just the single raw command.\n\n\
         Instruction: {}\n\
         Ensure the command is compatible with the specified OS and Shell. \
         Do not include any markdown formatting, backticks, or explanations.",
        os, shell, cwd, last_command, refinement
    )
}

/// Builds the prompt for explaining a command.
pub fn build_explain_prompt(command: &str) -> String {
    let os = std::env::consts::OS;
    format!(
        "Context:\n- Operating System: {}\n\n\
         Task: Provide a concise, high-signal technical breakdown of this command. \
         Assume the user is an expert CLI user. Focus on flag functions and non-obvious behavior. \
         No introductory fluff or basic definitions. \
         Command: {}",
        os, command
    )
}

/// Builds the prompt for diagnosing and fixing a command given error output.
pub fn build_fix_prompt(error_output: &str) -> String {
    let (os, shell, cwd) = get_env_context();
    format!(
        "Context:\n- Operating System: {}\n- Shell: {}\n- Current Directory: {}\n\n\
         Task: The following error occurred. \
         1. Shortly explain what went wrong (max 2 sentences). \
         2. Provide the fixed raw command line command. \
         Ensure the command is compatible with the specified OS and Shell.\n\n\
         Error Output:\n{}\n\n\
         Format your response exactly like this:\n\
         Explanation: [Your short explanation]\n\
         Command: [The raw command]",
        os, shell, cwd, error_output
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_env_context_non_empty() {
        let (os, shell, cwd) = get_env_context();
        assert!(!os.is_empty());
        assert!(!shell.is_empty());
        assert!(!cwd.is_empty());
    }

    #[test]
    fn test_parse_linux_distro() {
        let double_quoted = "NAME=\"Ubuntu\"\nVERSION=\"22.04.1 LTS\"\nPRETTY_NAME=\"Ubuntu 22.04.1 LTS\"\nID=ubuntu\n";
        assert_eq!(
            parse_linux_distro(double_quoted),
            Some("Ubuntu 22.04.1 LTS".to_string())
        );

        let single_quoted = "PRETTY_NAME='Arch Linux'\nID=arch\n";
        assert_eq!(
            parse_linux_distro(single_quoted),
            Some("Arch Linux".to_string())
        );

        let unquoted = "PRETTY_NAME=Fedora Linux\n";
        assert_eq!(
            parse_linux_distro(unquoted),
            Some("Fedora Linux".to_string())
        );

        let interior_quotes = "PRETTY_NAME=\"Distro 'Special' Edition\"\n";
        assert_eq!(
            parse_linux_distro(interior_quotes),
            Some("Distro 'Special' Edition".to_string())
        );

        let no_pretty = "NAME=\"Custom\"\nID=custom\n";
        assert_eq!(parse_linux_distro(no_pretty), None);
    }

    #[test]
    fn test_build_command_prompt() {
        let prompt = build_command_prompt("list all docker containers");
        assert!(prompt.contains("Context:"));
        assert!(prompt.contains("Operating System:"));
        assert!(prompt.contains("Shell:"));
        assert!(prompt.contains("Current Directory:"));
        assert!(prompt.contains("Query: list all docker containers"));
        assert!(prompt.contains("Do not include any markdown formatting"));
    }

    #[test]
    fn test_build_refine_prompt() {
        let prompt = build_refine_prompt("docker ps", "show stopped containers too");
        assert!(prompt.contains("Previous Command: docker ps"));
        assert!(prompt.contains("Instruction: show stopped containers too"));
        assert!(prompt.contains("Context:"));
    }

    #[test]
    fn test_build_explain_prompt() {
        let prompt = build_explain_prompt("tar -czvf archive.tar.gz /path");
        assert!(prompt.contains("Operating System:"));
        assert!(prompt.contains("Command: tar -czvf archive.tar.gz /path"));
        assert!(prompt.contains("high-signal technical breakdown"));
    }

    #[test]
    fn test_build_fix_prompt() {
        let prompt = build_fix_prompt("bash: foo: command not found");
        assert!(prompt.contains("Error Output:\nbash: foo: command not found"));
        assert!(prompt.contains("Explanation: [Your short explanation]"));
        assert!(prompt.contains("Command: [The raw command]"));
    }
}
