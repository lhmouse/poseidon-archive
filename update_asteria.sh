#!/bin/bash -e

_submodule="asteria"
_branch="master"
_message="submodule/${_submodule}: Update to current ${_branch}"

# Update to target branch
git submodule update --init -- "${_submodule}"
pushd "${_submodule}"
git checkout -f "${_branch}"
git pull --ff-only
popd

# Commit if there are diffs
if ! git diff HEAD --exit-code -- "${_submodule}"
then
  git commit -sm "${_message}" -- "${_submodule}"
fi
