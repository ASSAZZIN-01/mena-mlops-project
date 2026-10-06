#!/bin/sh
set -eu

: "${STABLE_WEIGHT:=95}"
: "${CANDIDATE_WEIGHT:=5}"

export STABLE_WEIGHT CANDIDATE_WEIGHT
envsubst '${STABLE_WEIGHT} ${CANDIDATE_WEIGHT}' \
  < /etc/nginx/templates/nginx.conf.template \
  > /etc/nginx/conf.d/default.conf

if [ "$STABLE_WEIGHT" -eq 0 ]; then
    sed -i '/server model-stable:/d' /etc/nginx/conf.d/default.conf
fi
if [ "$CANDIDATE_WEIGHT" -eq 0 ]; then
    sed -i '/server model-candidate:/d' /etc/nginx/conf.d/default.conf
fi

exec nginx -g 'daemon off;'
