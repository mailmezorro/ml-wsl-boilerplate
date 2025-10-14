#!/bin/bash
# -----------------------------------------------------------------------------
# MINIMAL INFRASTRUCTURE BOILERPLATE SETUP (FOR USERS WITH CONDA/MAMBA INSTALLED)
# Purpose: Create Mamba environment, install pre-commit hooks, and launch VS Code.
# USAGE: Simply execute ./setup.sh
# -----------------------------------------------------------------------------

# --- CONFIGURATION ---
ENV_NAME="ml-wsl-base"
ENV_FILE="environment.yml"

# SET TO 'true' TO AUTOMATICALLY INSTALL ZSH, OH-MY-ZSH, AND POWERLEVEL10K THEME.
# NOTE: This requires sudo permission for package installation (e.g., apt install zsh).
INSTALL_SHELL_THEME="true"

echo "================================================="
echo "STARTING INFRASTRUCTURE SETUP FOR $ENV_NAME"
echo "================================================="

# --- 1. PREREQUISITE CHECK (MAMBA/CONDA) ---
if command -v mamba &> /dev/null; then CONDA_CMD="mamba"; else CONDA_CMD="conda"; fi

if ! command -v $CONDA_CMD &> /dev/null; then
    echo "================================================="
    echo "ERROR: Neither 'mamba' nor 'conda' found."
    echo "Please install Miniforge/Mambaforge first."
    echo "================================================="
    exit 1
fi
echo "  > Using package manager: $CONDA_CMD"


# --- 2. CREATE OR UPDATE ENVIRONMENT ---
echo "--- 2. Environment '$ENV_NAME' is being created/updated ---"
$CONDA_CMD env create -f $ENV_FILE --yes || $CONDA_CMD env update -f $ENV_FILE --yes
if [ $? -ne 0 ]; then echo "ERROR: Environment creation/update failed."; exit 1; fi


# --- 3. CREATE BASE STRUCTURE & INSTALL HOOKS ---
echo "--- 3. Creating Base Structure and Installing Hooks ---"
[ ! -d 'data' ] && mkdir data && echo '  > Data folder created.'

# 3.1. CREATE RUFF CONFIGURATION
echo "  > Creating/Updating .pre-commit-config.yaml for Ruff (Linter & Formatter)..."
cat << EOF > .pre-commit-config.yaml
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.1.7 
  hooks:
    - id: ruff
      args: [--fix, --exit-non-zero-on-fix]
    - id: ruff-format
EOF
echo "  > .pre-commit-config.yaml created."

# 3.2. HOOK INSTALLATION
echo "  > Installing Git Pre-Commit Hooks (Ruff)..."
$CONDA_CMD run -n $ENV_NAME pre-commit install
if [ $? -ne 0 ]; then echo "WARNING: Pre-commit installation failed. Check if 'pre-commit' and 'ruff' are in $ENV_FILE."; exit 1; fi


# --- 4. VS CODE INTEGRATION ---
echo "--- 4. Configuring VS Code Extensions & Launching IDE ---"
if command -v code &> /dev/null
then
    echo "  > Installing/Updating recommended Extensions..."
    code --install-extension ms-python.python --force 2>/dev/null
    code --install-extension ms-toolsai.jupyter --force 2>/dev/null
    
    echo "SETUP SUCCESSFUL. Opening VS Code. Select environment '$ENV_NAME'."
    code .
else
    echo "WARNING: 'code' CLI not found. Please start VS Code manually and select environment '$ENV_NAME'."
fi


# --- 4.5. Create VS Code recommendations (Extensions.json) ---
VSCODE_DIR=".vscode"
VSCODE_EXT_FILE="$VSCODE_DIR/extensions.json"

echo "--- 4.5. Creating VS Code extension recommendations file ---"

if [ ! -d "$VSCODE_DIR" ]; then mkdir "$VSCODE_DIR"; fi

cat << EOF > "$VSCODE_EXT_FILE"
{
    "recommendations": [
        "ms-python.python",       
        "ms-toolsai.jupyter",      
        "eamodio.gitlens",         
        "charliermarsh.ruff",
        "ms-toolsai.datawrangler",
        "hediet.vscode-drawio"
    ]
}
EOF
echo "  > File $VSCODE_EXT_FILE created. VS Code will suggest other helpful extensions."

# -----------------------------------------------------------------------------
# 5. OPTIONAL SHELL ENHANCEMENT (ZSH + POWERLEVEL10K)
# -----------------------------------------------------------------------------
if [ "$INSTALL_SHELL_THEME" = "true" ]; then
    echo "--- 5. Installing Zsh and Powerlevel10k Theme (Advanced Shell) ---"

    # 5.1. Install Zsh (Requires sudo password for package manager) - Improved
    if ! command -v zsh &> /dev/null; then
        echo "  > Installing Zsh (requires sudo password)..."
        
        if command -v apt &> /dev/null; then
            sudo apt update && sudo apt install -y zsh
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y zsh
        else
            echo "ERROR: Unknown package manager. Please install zsh manually. Skipping shell theme."
        fi
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Zsh installation failed. Please check network/permissions. Skipping shell theme."
        fi
    else
        echo "  > Zsh is already installed."
    fi

    # Proceed only if zsh is available
    if command -v zsh &> /dev/null; then
        # 5.2. Install Oh My Zsh non-interactively
        if [ ! -d "$HOME/.oh-my-zsh" ]; then
            echo "  > Installing Oh My Zsh..."
            sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended > /dev/null 2>&1
        else
            echo "  > Oh My Zsh is already installed."
        fi

        # 5.3. Install Powerlevel10k Theme
        ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
        if [ ! -d "${ZSH_CUSTOM}/themes/powerlevel10k" ]; then
            echo "  > Installing Powerlevel10k theme..."
            git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "${ZSH_CUSTOM}/themes/powerlevel10k"
        fi

        # 5.4. Configure Zsh to use P10k
        ZSHRC_FILE="$HOME/.zshrc"
        if [ -f "$ZSHRC_FILE" ]; then
            echo "  > Setting Powerlevel10k as Zsh theme in $ZSHRC_FILE..."
            sed -i 's/ZSH_THEME="[^"]*"/ZSH_THEME="powerlevel10k\/powerlevel10k"/' "$ZSHRC_FILE"
            if ! grep -q 'powerlevel10k' "$ZSHRC_FILE"; then
                echo 'ZSH_THEME="powerlevel10k/powerlevel10k"' >> "$ZSHRC_FILE"
            fi
            
            # Ensure mamba shell init is run for zsh too
            echo "  > Initializing Mamba hook for Zsh..."
            $CONDA_CMD shell init --shell zsh > /dev/null 2>&1
            
            # 5.6. Conda initialization for Bash (if not already done)
            if ! grep -q "$CONDA_CMD shell hook" "$HOME/.bashrc"; then
                echo "  > Initializing Mamba hook for Bash..."
                "$CONDA_CMD" init bash > /dev/null 2>&1
            fi
        fi

        # 5.5. Instructions for the user (Manual Interactive Steps)
        echo ""
        echo "================================================="
        echo "ADVANCED SHELL SETUP IS COMPLETE (FINAL STEPS REQUIRED)"
        echo "1. Change your default shell: chsh -s \$(which zsh)"
        echo "2. **CLOSE AND REOPEN** your terminal to load Zsh."
        echo "3. Run 'p10k configure' to start the interactive wizard."
        echo "================================================="
    fi
fi

# 6. CLEANUP 
INSTALLER_FILE="Miniforge3-Linux-x86_64.sh"
if [ -f "$INSTALLER_FILE" ]; then
    echo "--- 6. Cleaning up installer file ---"
    rm "$INSTALLER_FILE"
    echo "  > Deleted installer file: $INSTALLER_FILE"
fi

echo "================================================="
exit 0