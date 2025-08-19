#!/bin/bash

java -jar src/progex_parser/progex-v3.4.5/progex.jar \
  -cfg -lang java \
  -format json \
  -outdir data/graphs \
  $1

# Usage: src/progex_parser/java_cfg.sh <path_to_java_file>
# Example: src/progex_parser/java_cfg.sh /path/to/your/JavaFile.java
