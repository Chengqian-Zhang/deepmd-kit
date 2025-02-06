for branch in $(git for-each-ref --format='%(refname:short)' refs/remotes); do
  if git show "$branch:deepmd/pt/loss/ep.py" &>/dev/null; then
    echo "文件存在于分支: $branch"
  fi
done
