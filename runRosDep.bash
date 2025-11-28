#!/usr/bin/env bash


rosdep update && \
  rosdep install --from-paths src --ignore-src -y

