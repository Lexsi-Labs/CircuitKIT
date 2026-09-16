# Lexsi Labs Source Available License (LSAL), Version 1.2 (CircuitKit)

## Preamble

This Source Available License governs use of the software known as **CircuitKit**, together with its example datasets, notebooks, and documentation (collectively, the "Licensed Work"), developed and owned by **Lexsi Labs (Lithasa Technologies Pvt. Ltd.)** ("Licensor").

This is **not** an open-source license as defined by the [Open Source Initiative (OSI)](https://opensource.org/). It grants **academic research and teaching** MIT-like permissions: free use, modification, and redistribution with the notice intact. Organizations must acknowledge their use or obtain permission (Section 1A). It also bars commercial exploitation and unsafe use. The Licensed Work locates and manipulates the internal circuits of language models, including safety-relevant behaviors, and its value lies in reproducible research and auditing. The restrictions exist to keep that work out of unreviewed production deployment and out of the removal of model safeguards.

---

## 1. Grant of Rights

Subject to the terms of this License, permission is hereby granted, free of charge, to any person obtaining a copy of the Licensed Work, to use, copy, modify, merge, publish, and redistribute the Licensed Work and derivative works thereof, **for Noncommercial Purposes only**, provided that the above copyright notice, this License, and the Responsible Use conditions (Section 4) are included in full in all copies or substantial portions of the Licensed Work.

For academic research and teaching, this grant is MIT-like: you may use, modify, and redistribute the Licensed Work, provided the copyright notice, this License, and Section 4 travel with every copy.

**Noncommercial Purposes** means:

* personal use for research, experimentation, private study, or hobby projects;
* academic and scholarly research, teaching, and publication (including use in papers, theses, benchmarks, and reproducibility artifacts).

Research and teaching by academics and their groups, including work done at a university or public research body, falls under this Section. Section 1A governs use by an organization as such, including a charitable organization, educational institution, public research organization, or government body acting for its own operations.

## 1A. Use by Organizations

Before any organization, including a charitable organization, educational institution, public research organization, or government body, uses the Licensed Work for **internal evaluation, red-teaming, benchmarking, safety auditing, or any use in connection with developing, evaluating, repairing, or safeguarding its own models, products, or services**, whether or not it charges a fee or provides the Licensed Work to third parties, it must either:

* obtain Licensor's **prior written permission**; or
* provide Licensor a written **acknowledgement of use**, identifying the organization, the intended use, and an undertaking to attribute the Licensed Work, which Licensor may accept in place of negotiated permission.

Both go to **support@lexsi.ai**.

Licensor grants permission at its discretion. The written permission sets out the terms, which may include sharing with Licensor the evaluation results the Licensed Work produces, in CircuitKit's standard report format, excluding model weights, training data, prompt text, and model generations. Licensor uses shared results for research and for improving the Licensed Work, and handles them under the terms stated in the written permission.

Permission under this Section does not authorize any use described in Section 2.

## 2. Commercial Restriction

Without a separate **commercial license** from Lexsi Labs, you may **not** Sell the Licensed Work. "**Sell**" means practicing any right granted to you under this License to provide to third parties, for a fee or other consideration, a product or service whose value derives, entirely or substantially, from the functionality of the Licensed Work, including:

* offering the Licensed Work, or a derivative of it, as a commercial product, paid service, SaaS, hosted, or API offering;
* embedding the Licensed Work in proprietary or revenue-generating software;
* paid consulting or support whose substance is the Licensed Work.

If you redistribute a modified version, you must mark it as modified and must not present it as the original. You may also not **re-license, rebrand, or redistribute** the Licensed Work under different terms, nor use **Lexsi Labs**, **CircuitKit**, or related trademarks, logos, or branding except to identify unmodified, licensed copies.

## 3. Patents

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. Any patent rights of Licensor relating to the Licensed Work are reserved and may be licensed only under a separate written agreement with Licensor.

## 4. Responsible Use (Safety Conditions)

The Licensed Work can locate the circuits that implement safety-relevant behaviors (such as refusal) and can modify models through them. As a condition of this License, you may **not**:

* use the Licensed Work to locate, remove, weaken, or bypass the safety behaviors of any model in order to deploy, distribute, or operationalize the resulting model outside a research, evaluation, or audit context;
* deploy a checkpoint exported by the Licensed Work's intervention modules in a production, user-facing, or agentic system without re-evaluating its safety-relevant behavior after the intervention;
* use the Licensed Work to intentionally produce, disseminate, or operationalize harmful model behavior outside a research, evaluation, or audit context.

## 4A. Third-Party Components

The Licensed Work operates on third-party base models and draws on third-party benchmarks and datasets. Each carries its own license, and this License grants no rights under any of them. You must comply with those terms in addition to this License, including any restrictions a base-model license places on derivative models or checkpoints you produce.

## 5. Ownership

All rights, title, and interest in and to the Licensed Work remain with **Lithasa Technologies Pvt. Ltd.** Except as expressly stated in Sections 1 and 1A, nothing in this License transfers ownership or any other rights to the Licensee.

## 6. Contributions

If you submit modifications, pull requests, or patches ("Contributions") to Lexsi Labs, you grant Lexsi Labs a perpetual, worldwide, royalty-free right to use, modify, distribute, and license your Contributions under any terms, including commercial ones, and you represent that you have the right to make such Contributions.

## 7. Warranty Disclaimer and Responsibility for Use

THE LICENSED WORK IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE LICENSOR OR CONTRIBUTORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE LICENSED WORK OR THE USE OR OTHER DEALINGS IN THE LICENSED WORK.

**You alone are responsible for your use of the Licensed Work.** This includes every model, checkpoint, output, evaluation, or system you produce with it, and your compliance with applicable law and with the Responsible Use conditions in Section 4. This responsibility applies to academic, personal, organizational, and commercial use alike. Licensor has no duty to monitor your use and bears no liability for misuse of the Licensed Work by you or by any third party who obtained it through you.

The Licensed Work is a research tool. Any measurements or safety-relevant findings it reports hold only under the conditions you configure. They do not certify that any model is safe to deploy; that decision rests with you.

If you use the Licensed Work under Section 1A or under a commercial license, you will, to the extent permitted by applicable law, indemnify and hold harmless Licensor and its contributors against any claim, loss, or expense, including legal fees, arising from your use or misuse of the Licensed Work or from any breach of this License by you. This indemnity does not apply to personal or academic use under Section 1; the two preceding paragraphs govern that use.

## 8. Termination

This License terminates without notice if you breach any of its terms. On termination, you must stop using the Licensed Work and destroy every copy in your possession. Licenses of parties who received the Licensed Work from you remain in force provided they remain in compliance.

Sections 3, 4A, 5, 6, 7, and 9 survive termination.

## 9. Governing Law and General Terms

This License shall be governed by and construed in accordance with the **laws of India**, without regard to its conflict of law principles. If a court holds any provision of this License unenforceable, the remaining provisions stay in effect. Licensor may publish revised versions of this License; a copy of the Licensed Work remains under the version it was received under unless you accept a later version.

## 10. Contact

For acknowledgements and permission requests under Section 1A, and for commercial use, partnership, or redistribution rights under Section 2, contact:
**support@lexsi.ai** · **https://lexsi.ai**

## 11. Notice

**CircuitKit** © 2026 **Lithasa Technologies Pvt. Ltd.**
Licensed under the **Lexsi Labs Source Available License (LSAL) v1.2**.
**Academic research and teaching are free on MIT-like terms. Use by any organization requires written acknowledgement or permission (Section 1A). Commercial use requires a commercial license (Section 2). The Licensed Work may not be used to remove or weaken the safety behaviors of deployed models (Section 4).**
