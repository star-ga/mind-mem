# Changelog Format Guide

## Format

mind-mem follows [Keep a Changelog](https://keepachangelog.com/) format.

### Version Header

```markdown
## [1.7.1] - 2026-02-24
```

### Change Categories

Use these categories in order:

1. **Added** — New features
2. **Changed** — Changes to existing features
3. **Deprecated** — Features that will be removed
4. **Removed** — Removed features
5. **Fixed** — Bug fixes
6. **Security** — Security fixes

### Entry Format

- Start with a verb (Add, Fix, Update, Remove)
- Reference issue numbers when applicable
- Keep entries concise (one line preferred)

### Example

```markdown
## [1.7.1] - 2026-02-24

### Added
- Add edge case tests for recall engine (#128)
- Add CodeQL security analysis workflow (#140)

### Fixed
- Fix tomllib import for Python 3.10 (#126)
- Fix benchmark recall() argument order (#127)
```
