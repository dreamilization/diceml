# DiceML project rules

## Git workflow

- Use `main` for completed versions. Develop each version on exactly one branch, named like `1.0-feat` (Git branch names cannot contain spaces).
- Keep all work for a version on its existing branch. Merge into `main` once that version is complete and verified.
- Use commit messages in the form `<version> <short description>`.
- Commit only as `dreamilization <40374124+dreamilization@users.noreply.github.com>`. Do not add assistant attribution or coauthor trailers.
- The repository remote is `https://github.com/dreamilization/diceml.git`.

## Product and design

- Build a native iPhone and iPad app that predicts the next ordered result for a configurable number of six-sided dice; default to three dice.
- Let users record actual rolls manually or through camera capture. Save entered history locally across launches and train on the device.
- Train a sequence model using previous actual rolls as inputs and the next roll's individual die values as targets. Predictions must use the learned model when history exists. Use random generation only when no input history exists.
- Keep predictions separate from actual observed outcomes; never train on predictions as if they were observations.
- Use an Apple Card inspired layout: a prominent prediction card shows the next dice values and their sum, with the user's actual previous rolls listed below.
- Avoid secondary titles and explanatory subtitles. Let layout and visual hierarchy explain the interface. Apply this rule throughout the design.
- Evaluate predictions chronologically against simple baselines. Do not imply that a model can reliably predict independent fair dice.
