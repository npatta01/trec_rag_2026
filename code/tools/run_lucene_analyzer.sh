#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
shared_root="$(dirname "$common_dir")"
cache_dir="${LUCENE_ANALYZER_CACHE_DIR:-$shared_root/cache/lucene-analyzer/10.4.0}"
source_dir="$repo_root/code/tools/lucene_analyzer"
image="${LUCENE_ANALYZER_JAVA_IMAGE:-docker.io/library/eclipse-temurin@sha256:1eeacc8c295ed4805f6ffead2417b1936aad296b02ea9e56b457230befc9e98d}"
container_name="${LUCENE_ANALYZER_CONTAINER:-trec-rag-lucene-analyzer}"
port="${LUCENE_ANALYZER_PORT:-18081}"
index_id="${LUCENE_ANALYZER_INDEX_ID:-hosted_climbmix_unknown_revision}"

mkdir -p "$cache_dir/lib" "$cache_dir/classes"

download_jar() {
  local artifact="$1"
  local jar="$cache_dir/lib/$artifact-10.4.0.jar"
  local base="https://repo1.maven.org/maven2/org/apache/lucene/$artifact/10.4.0/$artifact-10.4.0.jar"
  if [[ ! -s "$jar" ]]; then
    curl -fL --retry 3 --retry-delay 2 -o "$jar.partial" "$base"
    curl -fL --retry 3 --retry-delay 2 -o "$jar.sha512" "$base.sha512"
    expected="$(awk '{print $1}' "$jar.sha512")"
    actual="$(sha512sum "$jar.partial" | awk '{print $1}')"
    [[ "$expected" == "$actual" ]] || { echo "checksum mismatch for $artifact" >&2; exit 1; }
    mv "$jar.partial" "$jar"
  fi
}

download_jar lucene-core
download_jar lucene-analysis-common

podman image exists "$image" || podman pull "$image" >/dev/null
podman run --rm \
  -v "$source_dir:/src:ro" \
  -v "$cache_dir:/build:rw" \
  "$image" \
  javac -encoding UTF-8 -cp '/build/lib/*' -d /build/classes /src/AnalyzerServer.java

podman rm -f "$container_name" >/dev/null 2>&1 || true
podman run -d --name "$container_name" \
  -p "127.0.0.1:$port:$port" \
  -e "ANALYZER_PORT=$port" \
  -e "ANALYZER_INDEX_ID=$index_id" \
  -v "$cache_dir:/build:ro" \
  "$image" \
  java -cp '/build/classes:/build/lib/*' AnalyzerServer >/dev/null

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    curl -fsS "http://127.0.0.1:$port/health"
    printf '\n'
    exit 0
  fi
  sleep 1
done

podman logs "$container_name" >&2
exit 1
