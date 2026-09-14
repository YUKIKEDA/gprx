## Summary

<!-- What and why. Japanese is OK. -->

## Related

<!-- `Closes #N` when this PR finishes that Issue. N/A if none. -->

## Test plan

<!-- Commands and cases. -->

- [ ] `just lint`
- [ ] `just test`
- [ ] `just bench` if this PR touches a hot path (`kernel/`, workspace, gpr, objective, online)

## Verification

<!-- What you actually ran. N/A if not yet. Hot-path PRs: paste criterion vs the last named baseline (phase-1a / phase-1b). -->

## Risk / Rollback

N/A

## Checklist

- [ ] Matches `AGENTS.md`, `.cursor/rules/`, and `.dev/roadmap.md`
- [ ] No `unwrap` / `expect` / `panic` on library paths
- [ ] rustdoc in English for new public API
