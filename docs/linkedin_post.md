# LinkedIn post

**Image:** `docs/assets/linkedin_poster.png` (portrait 4:5, 2160 x 2700, the format that fills the most of a phone
screen in the feed), rebuilt by `scripts/make_linkedin_poster.py` from `paper/numbers.json` and the project's own code
on the synthetic data.

**Alt text for the image:** Poster for the research program "From Liaison Notes to Regional Indicators". The Bank of
Thailand hears from firms in every region four times a year, while official regional output arrives once a year. A
masked interview note (translated from Thai) is coded into topics (jobs, sales, costs, investment, outlook); 7,560 notes become a
regional activity index that tracks true activity through the 2020 COVID shock; an AI assistant drafts a report
sentence with a citation to a data table, and an injected error with a firm name, a made-up number and a bad citation
is blocked. Headline numbers: 0 names left after masking, correlation 0.89 to 0.92 after composition adjustment, 41%
lower nowcast error, power 0.92 for a 30-economist trial. All results are from synthetic data.

Every number below is in the paper and on the poster. Post the links as the first comment.

---

## Post (English, about 2,600 characters)

Four times a year, the Bank of Thailand hears directly from firms in every region.

Official regional output arrives once a year, and late.

The interviews are timely, rich and explain *why* things change. But they arrive as free Thai text, written by many
interviewers, from a panel of firms that changes every round. So most of their value ends up in a report someone
writes by hand.

I designed a research program around two questions:
1. Can those notes become reliable, timely regional indicators?
2. Can a generative-AI assistant cut the time to draft the regional report by more than 20% without lowering quality?

It is built as four studies:
• Study 1 codes every note on six topics (sales, jobs, investment, prices, costs, outlook) with a lexicon, a
fine-tuned Thai model and an LLM, scored against an expert gold standard, then builds diffusion indices weighted by
sector and corrected for the changing panel
• Study 2 tests whether the indices improve pseudo-real-time nowcasts (Clark–West, Diebold–Mariano)
• Study 3 builds an assistant that cites a source for every sentence and blocks any number it cannot trace to a data
table
• Study 4 is a pre-registered randomized crossover trial with economists drafting reports with and without it

Before touching confidential data, I built the whole pipeline in the open and ran it end to end on 7,560 synthetic
notes:

→ 0 firm or contact names left after masking
→ correcting for panel composition raises the index's correlation with true activity from 0.89 to 0.92
→ adding the index cuts nowcast error by 41% against an autoregressive benchmark
→ every drafted sentence is cited, and an injected error (a firm name, a made-up number, a bad citation) is blocked
→ 30 economists drafting four reports each give a power of 0.92 to detect a 35% time saving

The point I care most about: an AI that drafts central bank reports must be auditable sentence by sentence. Every
number checked mechanically, every statement traceable to a note or a table, nothing identifying ever in a draft.

Everything is open: the paper (also as a web page), code that rebuilds every figure and number with one command, the
synthetic dataset on Hugging Face, and a DOI on Zenodo. Links in the first comment.

Caveat up front: every result above comes from synthetic data. It shows the software works, not that the hypotheses
hold. The real notes are confidential and stay inside the Bank of Thailand.

If you work with liaison or survey text, regional nowcasting, or AI in official reporting, I would value your
critique. What would you test first?

Views are my own.

#CentralBanking #Nowcasting #TextAsData #GenerativeAI #LLM #RAG #Thailand #EconomicResearch

---

## First comment (links)

Paper (read online): https://thanatchon36.github.io/blp-research/paper/
Code: https://github.com/thanatchon36/blp-research
Data: https://huggingface.co/datasets/thanatchon36/blp-research
DOI: https://doi.org/10.5281/zenodo.24102338

---

## Post (Thai)

ปีละ 4 ครั้ง ธนาคารแห่งประเทศไทยได้ฟังเสียงผู้ประกอบการโดยตรงจากทุกภูมิภาค

แต่ตัวเลขเศรษฐกิจรายภาคอย่างเป็นทางการ ออกปีละครั้ง และออกช้า

บันทึกสัมภาษณ์เหล่านี้ทันเวลา ละเอียด และบอกได้ว่า "ทำไม" สถานการณ์จึงเปลี่ยน แต่ทั้งหมดเป็นข้อความภาษาไทยอิสระ
เขียนโดยผู้สัมภาษณ์หลายคน จากกลุ่มบริษัทที่เปลี่ยนไปทุกรอบ คุณค่าส่วนใหญ่จึงไปจบอยู่ในรายงานที่ต้องนั่งเขียนด้วยมือ

ผมออกแบบงานวิจัยขึ้นมาเพื่อตอบคำถามสองข้อ
1. บันทึกเหล่านี้กลายเป็นเครื่องชี้เศรษฐกิจรายภาคที่เชื่อถือได้และทันเวลาได้หรือไม่
2. ผู้ช่วย generative AI ช่วยลดเวลาร่างรายงานภาวะเศรษฐกิจภาคได้เกิน 20% โดยคุณภาพไม่ลดลงได้หรือไม่

งานวิจัยแบ่งเป็น 4 การศึกษา
• การศึกษาที่ 1 ลงรหัสบันทึกทุกฉบับใน 6 หัวข้อ (ยอดขาย การจ้างงาน การลงทุน ราคา ต้นทุน แนวโน้ม) ด้วย lexicon
โมเดลภาษาไทยที่ fine-tune และ LLM เทียบกับ gold standard ของผู้เชี่ยวชาญ แล้วสร้างดัชนีการแพร่กระจายที่ถ่วงน้ำหนักตาม
สาขาและปรับผลจากกลุ่มตัวอย่างที่เปลี่ยนไป
• การศึกษาที่ 2 ทดสอบว่าดัชนีช่วยให้ nowcast แบบ pseudo-real-time แม่นขึ้นหรือไม่ (Clark–West, Diebold–Mariano)
• การศึกษาที่ 3 สร้างผู้ช่วยที่อ้างอิงแหล่งที่มาทุกประโยค และบล็อกตัวเลขที่ตรวจย้อนกลับไปหาตารางข้อมูลไม่ได้
• การศึกษาที่ 4 เป็นการทดลองแบบสุ่มไขว้ (randomized crossover) ที่ลงทะเบียนแผนล่วงหน้า ให้นักเศรษฐศาสตร์ร่างรายงาน
ทั้งแบบมีและไม่มีผู้ช่วย

ก่อนแตะข้อมูลที่เป็นความลับ ผมสร้าง pipeline ทั้งหมดแบบเปิด และรันครบทุกขั้นบนบันทึกจำลอง 7,560 ฉบับ

→ ไม่เหลือชื่อบริษัทหรือผู้ให้ข้อมูลเลยหลังปิดบังข้อมูล
→ การปรับผลจากกลุ่มตัวอย่างที่เปลี่ยนไป ยกสหสัมพันธ์ระหว่างดัชนีกับภาวะเศรษฐกิจจริงจาก 0.89 เป็น 0.92
→ เพิ่มดัชนีเข้าไปแล้วความคลาดเคลื่อนของ nowcast ลดลง 41% เทียบกับแบบจำลอง autoregressive
→ ทุกประโยคที่ร่างมีการอ้างอิง และข้อผิดพลาดที่จงใจใส่เข้าไป (ชื่อบริษัท ตัวเลขปลอม อ้างอิงผิด) ถูกบล็อกทั้งหมด
→ นักเศรษฐศาสตร์ 30 คน ร่างคนละ 4 รายงาน ให้ power 0.92 ในการตรวจพบการลดเวลา 35%

สิ่งที่ผมให้ความสำคัญที่สุด: AI ที่ร่างรายงานของธนาคารกลางต้องตรวจสอบได้ทีละประโยค ตัวเลขทุกตัวถูกตรวจด้วยระบบ
ทุกข้อความย้อนกลับไปหาบันทึกหรือตารางได้ และไม่มีข้อมูลที่ระบุตัวตนหลุดไปในร่างเลย

ทุกอย่างเปิดหมด: paper (อ่านเป็นหน้าเว็บได้) โค้ดที่สร้างทุกกราฟและทุกตัวเลขใหม่ได้ด้วยคำสั่งเดียว ชุดข้อมูลจำลองบน
Hugging Face และ DOI บน Zenodo ลิงก์อยู่ในคอมเมนต์แรกครับ

ข้อควรระวัง: ผลทั้งหมดข้างต้นมาจากข้อมูลจำลอง แสดงว่าซอฟต์แวร์ทำงานได้ ไม่ได้แสดงว่าสมมติฐานเป็นจริง
บันทึกจริงเป็นความลับและอยู่ภายในธนาคารแห่งประเทศไทยเท่านั้น

ใครทำงานกับข้อความจากการสัมภาษณ์หรือแบบสำรวจ nowcasting รายภาค หรือ AI ในงานรายงานทางการ ยินดีรับคำวิจารณ์ครับ
ถ้าเป็นคุณ จะทดสอบอะไรก่อน?

ความเห็นเป็นของผมเอง

#CentralBanking #Nowcasting #TextAsData #GenerativeAI #LLM #RAG #Thailand #EconomicResearch

---

## Short version (English)

Four times a year, the Bank of Thailand hears from firms in every region. Official regional output arrives once a
year.

My new research program turns those Thai interview notes into regional indicators, and tests whether an AI
assistant that cites every sentence can cut report-drafting time by more than 20%.

On 7,560 synthetic notes, end to end:
• 0 names left after masking
• index–activity correlation 0.89 → 0.92 after composition adjustment
• 41% lower nowcast error than an AR benchmark
• every injected error blocked by the verifier

Synthetic data only: the software works; the hypotheses are still to be tested. Paper, code, data and DOI in the
first comment.

#CentralBanking #Nowcasting #GenerativeAI
