#!/bin/sh
set -eu

: "${STABLE_WEIGHT:=95}"
: "${CANDIDATE_WEIGHT:=5}"

export STABLE_WEIGHT CANDIDATE_WEIGHT
envsubst '${STABLE_WEIGHT} ${CANDIDATE_WEIGHT}' \
  < /etc/nginx/templates/nginx.conf.template \
  > /etc/nginx/conf.d/default.conf

exec nginx -g 'daemon off;'
