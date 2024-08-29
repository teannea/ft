_all:
  just --list

freeze:
  uv pip freeze | sed -e 's/\x1b\[[0-9;]*m//g' | uv pip compile - -o requirements.txt
