// report.html 의 $...$, $$...$$ 수식을 KaTeX로 렌더링한 뒤 Chromium으로 PDF를 만든다.
//   node build_report.js <katex node_modules 경로>
const fs = require('fs'), path = require('path');
const nm = process.argv[2];
const katex = require(path.join(nm, 'katex'));
const { chromium } = require('playwright');
let html = fs.readFileSync(path.join(__dirname, 'report.html'), 'utf8');
const render = (tex, display) => katex.renderToString(tex.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>'),
  { displayMode: display, throwOnError: true, strict: false });
html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => render(t, true));
html = html.replace(/\$([^$\n]+?)\$/g, (_, t) => render(t, false));
html = html.replace('<!--KATEX_CSS-->', `<link rel="stylesheet" href="file://${path.join(nm, 'katex/dist/katex.min.css')}">`);
const out = path.join(__dirname, '_report_rendered.html');
fs.writeFileSync(out, html);
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + out); await p.evaluate(() => document.fonts.ready);
  await p.pdf({ path: path.join(__dirname, 'EM2_TermProject_Report.pdf'), format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#777"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: '18mm', bottom: '18mm', left: '17mm', right: '17mm' } });
  await b.close(); fs.unlinkSync(out);
})();
