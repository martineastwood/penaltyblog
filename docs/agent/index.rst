Agent Support
=============

If you are using a coding agent such as Claude Code, Cursor, or Copilot, you can use the
``SKILL.md`` file to give your agent context about the penaltyblog public API.

Setup
-----

The skill file is included in the repository at
``.claude/skills/penaltyblog/SKILL.md``:

- `.claude/skills/penaltyblog/SKILL.md <https://github.com/martineastwood/penaltyblog/blob/master/.claude/skills/penaltyblog/SKILL.md>`_

The file is self-contained and covers all public modules: models, betting, matchflow,
implied odds, scrapers, FPL, ratings, backtest, visualization, and metrics.

No additional files are needed — your agent only needs this single file to understand the
package API and provide accurate code suggestions.
