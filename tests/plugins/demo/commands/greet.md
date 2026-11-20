---
description: Greet someone by name.
argument-hint: <who> [how]
disable-model-invocation: true
arguments:
- name: who
  description: Who to greet.
  required: true
- name: how
  description: How warmly.
  default: warmly
---
Greet {{ who }} {{ how }}. A flow's ${param} and {"json": 1} stay as written.
