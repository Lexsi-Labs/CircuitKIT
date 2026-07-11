# Lexsi Labs Source Available License (LSAL) – Version 1.1 (CircuitKit)

## Preamble

This Source Available License governs use of the software known as **CircuitKit**, together with its example datasets, notebooks, and documentation (collectively, the "Licensed Work"), developed and owned by **Lexsi Labs (Lithasa Technologies Pvt. Ltd.)** ("Licensor").

This is **not** an open-source license as defined by the [Open Source Initiative (OSI)](https://opensource.org/). It grants broad, free access to the source code and artifacts for **research, evaluation, education, and audit**, while restricting commercial exploitation and unsafe use. These restrictions are deliberate: the Licensed Work locates and manipulates the internal circuits of language models, including safety-relevant behaviors, and its value lies in reproducible research and auditing, not in unreviewed production deployment or the removal of model safeguards.

---

## 1. Grant of Rights

Subject to the terms of this License, permission is hereby granted, free of charge, to any person obtaining a copy of the Licensed Work, to use, copy, modify, merge, publish, and redistribute the Licensed Work and derivative works thereof, **for Noncommercial Purposes only**, provided that the above copyright notice, this License, and the Responsible Use conditions (Section 4) are included in full in all copies or substantial portions of the Licensed Work.

**Noncommercial Purposes** means:

* personal use for research, experimentation, private study, or hobby projects;
* academic and scholarly research, teaching, and publication (including use in papers, theses, benchmarks, and reproducibility artifacts);
* internal evaluation, red-teaming, or safety auditing that is not itself a paid product or service;
* use by charitable organizations, educational institutions, public research organizations, or government bodies for non-revenue-generating purposes.

## 2. Commercial Restriction

Without a separate **commercial license** from Lexsi Labs, you may **not** Sell the Licensed Work. "**Sell**" means practicing any right granted to you under this License to provide to third parties, for a fee or other consideration, a product or service whose value derives, entirely or substantially, from the functionality of the Licensed Work — including without limitation:

* offering the Licensed Work, or a derivative of it, as a commercial product, paid service, SaaS, hosted, or API offering;
* embedding the Licensed Work in proprietary or revenue-generating software;
* paid consulting or support whose substance is the Licensed Work.

You may also not **re-license, rebrand, or redistribute** the Licensed Work under different terms, nor use **Lexsi Labs**, **CircuitKit**, or related trademarks, logos, or branding except to identify unmodified, licensed copies.

## 3. Patents

The Licensor grants you a non-exclusive, worldwide, royalty-free patent license, under patent claims the Licensor can license that are necessarily infringed by the Licensed Work, to use the Licensed Work for Noncommercial Purposes as permitted by this License. This patent license terminates immediately if you or your company make any written claim that the Licensed Work infringes a patent.

## 4. Responsible Use (Safety Conditions)

The Licensed Work can locate the circuits that implement safety-relevant behaviors (such as refusal) and can modify models through them. As a condition of this License, you may **not**:

* use the Licensed Work to locate, remove, weaken, or bypass the safety behaviors of any model in order to deploy, distribute, or operationalize the resulting model outside a research, evaluation, or audit context;
* deploy a checkpoint exported by the Licensed Work's intervention modules in a production, user-facing, or agentic system without re-evaluating its safety-relevant behavior after the intervention;
* use the Licensed Work to intentionally produce, disseminate, or operationalize harmful model behavior outside a research, evaluation, or audit context.

## 5. Ownership

All rights, title, and interest in and to the Licensed Work remain with **Lithasa Technologies Pvt. Ltd.** Except as expressly stated in Sections 1 and 3, nothing in this License transfers ownership or any other rights to the Licensee.

## 6. Contributions

If you submit modifications, pull requests, or patches ("Contributions") to Lexsi Labs, you grant Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute, and license your Contributions under any terms, including commercial ones, and you represent that you have the right to make such Contributions.

## 7. Warranty Disclaimer

THE LICENSED WORK IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE LICENSOR OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE LICENSED WORK OR THE USE OR OTHER DEALINGS IN THE LICENSED WORK.

## 8. Termination

This License terminates automatically if you breach any of its terms. Upon termination, you must immediately cease use and destroy all copies of the Licensed Work in your possession. Licenses of parties who received the Licensed Work from you remain in force provided they remain in compliance.

## 9. Governing Law

This License shall be governed by and construed in accordance with the **laws of India**, without regard to its conflict of law principles.

## 10. Contact for Commercial Licensing

For commercial use, partnership, or redistribution rights, contact:
**support@lexsi.ai** · **https://lexsi.ai**

## 11. Notice

**CircuitKit** © 2026 **Lithasa Technologies Pvt. Ltd.**
Licensed under the **Lexsi Labs Source Available License (LSAL) v1.1**.
**Not for commercial use, and not for removing or weakening the safety behaviors of deployed models, without explicit permission.**
