# Security Policy

## Supported Versions

We provide security fixes for the latest release of CircuitKIT only. Please
upgrade to the latest version before reporting an issue.

| Version        | Supported          |
| -------------- | ------------------ |
| Latest release | :white_check_mark: |
| Older releases | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Report them privately with GitHub's private vulnerability reporting: open the
[Security tab](https://github.com/Lexsi-Labs/CircuitKIT/security) of this
repository and choose **Report a vulnerability**.

### What to Include

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact**
4. **Affected versions** (if known)
5. **Suggested fix** (if you have one)

### What to Expect

1. **Acknowledgment**: we confirm receipt of your report.
2. **Investigation**: we assess the severity and affected versions.
3. **Updates**: we keep you informed of our progress.
4. **Resolution**: we release a patch version and publish a GitHub Security
   Advisory once users have had time to update.
5. **Credit**: with your permission, we acknowledge your contribution.

## Security Best Practices

- Keep CircuitKIT and its dependencies up to date.
- Only load models, tokenizers and datasets from sources you trust.
- Never hardcode API keys or tokens; use environment variables.
- Run `pip-audit` to check your installation for known vulnerabilities.

## Third-Party Dependencies

CircuitKIT depends on third-party libraries such as PyTorch. We monitor
security advisories for them and raise version floors when a fix is available.

## Disclosure Policy

When a security issue is fixed we release a patch version, note it in the
CHANGELOG, and publish a GitHub Security Advisory.
