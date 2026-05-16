# Umbrella

## How automated changes are made

When changes are requested for this repository, the automation works in a local, sandboxed clone first. Files are edited locally and any existing lint/test commands are run. Only after those local updates are ready does the agent commit and push the changes to the existing GitHub pull request. The agent does not open a new PR by itself.
