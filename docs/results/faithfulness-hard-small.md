# Document faithfulness: hard-small

Model `claude-haiku`.

| measure | share |
|---|---|
| listed facts stated | 99.4% (326/328, 98%–100%) |
| listed facts judged wrong | 0.0% (0/328, 0%–1%) |
| listed facts judged missing | 0.6% (2/328, 0%–2%) |
| documents with an unlisted claim | 2.8% (4/145, 1%–7%) |
| control: removed fact called MISSING | 93.3% (14/15, 70%–99%) |
| control: reversed fact called WRONG | 0.0% (0/15, 0%–20%) |

By code, no model: of 290 sentences that state one fact, 0 name the object before the subject and 6 do not name the subject in the sentence.

## Unlisted claims the judge reported

- D00000: a product that buyers often compare with similar equipment
- D00050: The same person has also worked for Luse Group and Suvu Doveku
- D00067: which has drawn attention from several buyers
- D00130: Life there moves with the tides and the seasons, which shapes how people plan their weeks
