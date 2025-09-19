#!/bin/bash
# setup.sh - Download the repo and install dependencies

source ./config.sh

# Function to search upwards for a .env file
find_dotenv_upwards() {
  CURRENT_DIR=$(pwd)
  while [ "$CURRENT_DIR" != "/" ]; do
    if [ -f "$CURRENT_DIR/.env" ]; then
      echo "$CURRENT_DIR/.env"
      return 0
    fi
    CURRENT_DIR=$(dirname "$CURRENT_DIR")
  done
  return 1
}

download_or_pull_repo() {
  local repo_name=$1
  local repo_path=$2

  
  if [ -d "$repo_path/.git" ]; then
    echo "Repo exists at $repo_path. Pulling latest changes..."
    git -C "$repo_path" pull
  else
    echo "Cloning repo into $repo_path..."
    # Clone the repository using the token. Set both either as an env variable or in the .env
    git clone https://${GITHUB_TOKEN}@github.com/${repo_name}.git ${repo_path} || { echo "Git clone failed"; return 1; }
  fi

  # Change to the repo directory
  cd "$repo_path" || { echo "Failed to cd to $repo_path"; return 1; }

  # Install Python requirements
   if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt || { echo "pip install failed for $repo_name"; return 1; }
  else
    echo "No requirements.txt found in $repo_path"
  fi
}

# Try to find and source the .env
DOTENV_PATH=$(find_dotenv_upwards)
if [ -n "$DOTENV_PATH" ]; then
  echo "Loading environment from $DOTENV_PATH"
  set -a
  source "$DOTENV_PATH"
  set +a
else
  echo "No .env file found in parent directories."
fi

# Use a venv in the persistent volume so that the Python packages don't
# have to be reinstalled every container start
if [ -d "$VENV_PATH" ]; then
  echo "Virtual environment already exists at $VENV_PATH"
else
  echo "Creating virtual environment at $VENV_PATH"
  python3 -m venv --system-site-packages "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Set Jupyter password from env var if set
if [ -z "$JUPYTER_PASSWORD" ]; then
  echo "No Jupyter password set, skipping..."
  echo "Provide password as env variable JUPYTER_PASSWORD"
else
  echo "Set Jupyter password..."
  PASSWORD_HASH=$(python3 -c "from jupyter_server.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))")
  mkdir -p ~/.jupyter
  echo "{\"ServerApp\": {\"password\": \"$PASSWORD_HASH\"}}" > ~/.jupyter/jupyter_server_config.json
fi

# Clone or pull nano diffusion repo
download_or_pull_repo "$REPO_NAME" "$REPO_PATH"

# Go back to root dir to ensure consistency for running CMD
cd "/"

# Forward command passed from CMD or `docker run`
exec "$@"