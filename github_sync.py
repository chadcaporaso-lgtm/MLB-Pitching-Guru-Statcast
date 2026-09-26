import os
import json
import subprocess

def load_config():
    keys_path = "/content/drive/MyDrive/MLB-Guru-Data/api_keys.json"
    with open(keys_path, "r") as f:
        data = json.load(f)
    return data["GITHUB_USERNAME"], data["GITHUB_TOKEN"], data["GITHUB_REPO"]

def clone_or_pull(dest_dir="/content/MLB-Pitching-Guru-Statcast", branch="main"):
    user, token, repo = load_config()
    remote_url = f"https://{token}@github.com/{user}/{repo}.git"

    if os.path.exists(os.path.join(dest_dir, ".git")):
        print(f"[i] Pulling updates into {dest_dir}...")
        subprocess.run(["git", "-C", dest_dir, "pull", "origin", branch], check=True)
    else:
        print(f"[i] Cloning {repo} into {dest_dir}...")
        subprocess.run(["git", "clone", remote_url, dest_dir], check=True)
    
    subprocess.run(["git", "-C", dest_dir, "config", "user.name", user], check=True)
    subprocess.run(["git", "-C", dest_dir, "config", "user.email", f"{user}@users.noreply.github.com"], check=True)
    print(f"[✓] Synced: {dest_dir}")

def push_changes(commit_msg="Update pipeline from Colab", repo_dir="/content/MLB-Pitching-Guru-Statcast", branch="main"):
    user, token, repo = load_config()
    remote_url = f"https://{token}@github.com/{user}/{repo}.git"

    subprocess.run(["git", "-C", repo_dir, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_dir, "commit", "-m", commit_msg], check=False)
    subprocess.run(["git", "-C", repo_dir, "push", remote_url, branch], check=True)
    print(f"[✓] Pushed to {repo}:{branch}")
