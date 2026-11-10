#!/bin/sh

sanitize_runtime_env() {
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    old_ifs=$IFS
    IFS=':'
    set -- $LD_LIBRARY_PATH
    IFS=$old_ifs

    cleaned_ld=""
    for entry in "$@"; do
      [ -n "$entry" ] || continue
      case ":$cleaned_ld:" in
        *":$entry:"*) continue ;;
      esac
      cleaned_ld="${cleaned_ld:+$cleaned_ld:}$entry"
    done

    if [ -n "$cleaned_ld" ]; then
      export LD_LIBRARY_PATH="$cleaned_ld"
    else
      unset LD_LIBRARY_PATH
    fi
  fi

  export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
}
