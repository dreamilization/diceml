#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_dir"

remote_url='https://github.com/dreamilization/diceml.git'
version_branch='1.0-feat'

if [ ! -e .git ]; then
    git init -b main
fi

if [ "$(git rev-parse --show-toplevel)" != "$repo_dir" ]; then
    printf '%s\n' 'The Git repository does not match the DiceML directory.' >&2
    exit 1
fi

git config --local user.name 'dreamilization'
git config --local user.email '40374124+dreamilization@users.noreply.github.com'

if git remote get-url origin >/dev/null 2>&1; then
    if [ "$(git remote get-url origin)" != "$remote_url" ]; then
        printf '%s\n' 'origin already points to another repository; no remote was changed.' >&2
        exit 1
    fi
else
    git remote add origin "$remote_url"
fi

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    git branch -M main
    git add -- .gitignore AGENTS.md diceml diceml.xcodeproj scripts/initialize-git.sh
    git diff --cached --check
    GIT_AUTHOR_NAME='dreamilization' \
    GIT_AUTHOR_EMAIL='40374124+dreamilization@users.noreply.github.com' \
    GIT_COMMITTER_NAME='dreamilization' \
    GIT_COMMITTER_EMAIL='40374124+dreamilization@users.noreply.github.com' \
    git commit -m '1.0 initialize DiceML project'
fi

git push -u origin main

if git show-ref --verify --quiet "refs/heads/$version_branch"; then
    git switch "$version_branch"
else
    git switch -c "$version_branch"
fi
