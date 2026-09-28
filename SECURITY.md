# Security policy

## Supported versions

Security fixes target the latest published version of Aiython. Please upgrade
before reporting an issue with an older version.

## Report a vulnerability

Use [GitHub's private vulnerability reporting](https://github.com/sunmodza/aiython/security/advisories/new).
Include the affected version, impact, and steps to reproduce. Please do not
open a public issue for an undisclosed vulnerability or include API keys.

## Runtime permissions

Aiython is not a sandbox. AI tools use the permissions of the Aiython process,
and relevant code or data may be sent to the configured provider. Review the
script, provider, and available permissions before running it.
