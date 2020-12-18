# ARSENAL Base

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repo can be used as the foundation for building domain-specific instantiations of
ARSENAL. To create such an instantiation, it is recommended that this repo is
*forked*. This allows you to create pull requests in case of any improvements to
the base files, while keeping your domain-specific content separate.

This repo contains a simple end-to-end example of parsing regular expressions,
in the `./regexp` directory, which can form a starting point for other domains.

We also provide two additional entity processors that are not used by the
regexp example, but may be useful in other domain:
- a "no-op" entity processor (in `./noop-entity`)
- a mostly generic entity processor based on Spacy (in `./sal-entity`)

See the `./doc` directory for the complete documentation.

Arsenal is available under GPLv3. If GPLv3 does not fit your needs,
please contact licensee-ops@sri.com to discuss other licensing options.
