# Code Review Skill

This skill helps Claude perform thorough code reviews on pull requests or local changes in the NeMo project.

## Usage

Invoke this skill when asked to review code, PRs, or changes:
- "Review this PR"
- "Do a code review of these changes"
- "Check this code for issues"
- "Review the diff"

## Process

### 1. Gather Context

```bash
# View the current diff
git diff HEAD~1..HEAD

# Or for a specific PR branch
git diff main...<branch-name>

# Get list of changed files
git diff --name-only HEAD~1..HEAD

# View full file context for changed files
git show HEAD:<filepath>
```

### 2. Review Checklist

When reviewing code, check for the following:

#### Correctness
- [ ] Logic errors or off-by-one mistakes
- [ ] Edge cases not handled (empty inputs, None values, boundary conditions)
- [ ] Incorrect tensor shapes or dtype mismatches in ML code
- [ ] Proper error handling and informative error messages
- [ ] Return values are correct and consistent

#### NeMo-Specific Concerns
- [ ] Config dataclasses use `@dataclass` with proper type hints
- [ ] Models register themselves correctly with the NeMo registry
- [ ] `save_to` / `restore_from` compatibility maintained
- [ ] Mixed precision (BF16/FP16) considerations addressed
- [ ] Megatron-LM parallelism (TP/PP/DP) correctness if applicable
- [ ] Lightning module hooks used correctly (`training_step`, `validation_step`, etc.)
- [ ] CUDA memory management (avoid unnecessary `.cpu()` calls in hot paths)

#### Code Quality
- [ ] Functions and classes have docstrings
- [ ] Variable names are descriptive
- [ ] No dead code or commented-out blocks left in
- [ ] No hardcoded magic numbers (use constants or config)
- [ ] DRY — no duplicated logic that should be refactored
- [ ] Type hints present on public APIs

#### Testing
- [ ] New functionality has corresponding unit tests
- [ ] Tests cover edge cases, not just the happy path
- [ ] Tests are deterministic (no random seeds left unset)
- [ ] Mocks/stubs used appropriately for external dependencies
- [ ] Test names clearly describe what is being tested

#### Performance
- [ ] No unnecessary data copies or type conversions
- [ ] Loops that could be vectorized with torch ops
- [ ] Avoid Python loops over batch dimensions
- [ ] Caching used where appropriate

#### Security & Safety
- [ ] No credentials or API keys hardcoded
- [ ] User inputs validated before use
- [ ] File paths sanitized if accepting external input

### 3. Provide Structured Feedback

Organize feedback into categories:

**Blocking Issues** — Must fix before merge:
- Bugs that would cause incorrect results or crashes
- Breaking changes to public APIs without deprecation
- Missing tests for critical functionality
- Security vulnerabilities

**Suggestions** — Recommended but not blocking:
- Performance improvements
- Code clarity improvements
- Additional test coverage
- Better error messages

**Nits** — Minor style or preference items:
- Whitespace or formatting inconsistencies
- Minor naming improvements
- Optional docstring additions

### 4. Example Review Comment Format

```
## Code Review Summary

**Overall:** [Approve / Request Changes / Comment]

### Blocking Issues
1. **[File:Line]** Description of the issue and why it's a problem.
   ```python
   # Suggested fix:
   corrected_code_here
   ```

### Suggestions
1. **[File:Line]** Description of suggestion and benefit.

### Nits
1. **[File:Line]** Minor note.

### Positive Notes
- What was done well in this PR.
```

## Tips

- Always read the PR description and linked issue before reviewing the diff
- Check if existing tests still pass conceptually with the new changes
- When in doubt about intent, ask a clarifying question rather than assuming
- Be constructive — explain *why* something is an issue, not just *what* is wrong
- Acknowledge good patterns and improvements, not just problems
