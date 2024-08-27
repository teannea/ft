_all:
  just --list

freeze:
  uv pip freeze --color never | uv pip compile - -o requirements.txt
