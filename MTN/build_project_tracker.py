from pathlib import Path
import json
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.worksheet.table import Table, TableStyleInfo

OUT = Path(r'D:\IDD-PeD-Evaluation\outputs\IDD_PeD_MTN_Complete_Project_Tracker.xlsx')
RESULTS = Path(r'D:\IDD-PeD-Evaluation\outputs\final_validation\final_four_way_results.json')
NAVY, BLUE, TEAL, ORANGE, RED, PALE = '102A43','2F80ED','00A6A6','F2994A','C0392B','EEF5FF'
THIN = Side(style='thin', color='D9E2EC')

def style(ws, title, subtitle=''):
    ws.sheet_view.showGridLines = False
    ws.sheet_view.zoomScale = 90
    ws.freeze_panes = 'A4'
    ws.merge_cells('A1:H1'); ws['A1'] = title; ws['A1'].font = Font(name='Aptos Display', size=20, bold=True, color='FFFFFF'); ws['A1'].fill = PatternFill('solid', fgColor=NAVY); ws['A1'].alignment = Alignment(vertical='center')
    ws.row_dimensions[1].height = 34
    ws.merge_cells('A2:H2'); ws['A2'] = subtitle; ws['A2'].font = Font(name='Aptos', size=10, italic=True, color='52616B'); ws['A2'].alignment = Alignment(wrap_text=True, vertical='center'); ws.row_dimensions[2].height = 30
    for col in range(1, 12): ws.column_dimensions[chr(64+col)].width = 20

def table(ws, start_row, headers, rows, name=None, widths=None):
    for c, h in enumerate(headers, 1):
        cell=ws.cell(start_row,c,h); cell.font=Font(name='Aptos',bold=True,color='FFFFFF'); cell.fill=PatternFill('solid',fgColor=NAVY); cell.alignment=Alignment(wrap_text=True,vertical='center',horizontal='center'); cell.border=Border(top=THIN,bottom=THIN,left=THIN,right=THIN)
    ws.row_dimensions[start_row].height=30
    for r, row in enumerate(rows, start_row+1):
        longest=max((len(str(v)) for v in row if v is not None), default=0)
        ws.row_dimensions[r].height=max(22, min(60, 15*(1+longest//55)))
        for c, val in enumerate(row,1):
            cell=ws.cell(r,c,val); cell.font=Font(name='Aptos',size=10,color='243B53'); cell.alignment=Alignment(wrap_text=True,vertical='top'); cell.border=Border(top=THIN,bottom=THIN,left=THIN,right=THIN)
            head=str(headers[c-1])
            if isinstance(val,(int,float)):
                if 'Parameters' in head: cell.number_format='#,##0'
                elif any(x in head for x in ('MSE','Latency','MiB')): cell.number_format='0.00'
            if r%2==0: cell.fill=PatternFill('solid',fgColor='F7FAFC')
    end=start_row+len(rows)
    ws.auto_filter.ref=f'A{start_row}:{chr(64+len(headers))}{end}'
    if name:
        t=Table(displayName=name,ref=f'A{start_row}:{chr(64+len(headers))}{end}'); t.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=True,showColumnStripes=False); ws.add_table(t)
    if widths:
        for i,w in enumerate(widths,1): ws.column_dimensions[chr(64+i)].width=w
    ws.auto_filter.ref=f'A{start_row}:{chr(64+len(headers))}{end}'
    return end

def note(ws, cell, text, color=TEAL):
    ws[cell]=text; ws[cell].font=Font(name='Aptos',size=10,bold=True,color='FFFFFF'); ws[cell].fill=PatternFill('solid',fgColor=color); ws[cell].alignment=Alignment(wrap_text=True,vertical='center'); ws[cell].border=Border(top=THIN,bottom=THIN,left=THIN,right=THIN)

def main():
    r=json.loads(RESULTS.read_text(encoding='utf-8'))['models']
    model_map={
        'Context-Aware MTN':'B',
        'Context-Aware MTN with Encoder Self-Attention Sharing':'C4',
        'MTN with Dynamic Multimodal Fusion':'F',
        'Dynamic Multimodal Fusion MTN with Encoder Self-Attention Sharing':'F_C4'}
    wb=Workbook(); wb.remove(wb.active)
    # computed, saved unified values
    unified=[]; seedrows=[]
    for display,key in model_map.items():
        runs=r[key]; vals={m:[runs[str(s)]['metrics'][m] for s in (20260928,20260929,20260930)] for m in ('mse','cmse','cfmse','seconds')}
        import statistics
        p=runs['20260928']['parameters']; ck=runs['20260928']['checkpoint_bytes']/1048576
        unified.append([display,statistics.mean(vals['mse']),statistics.stdev(vals['mse']),statistics.mean(vals['cmse']),statistics.stdev(vals['cmse']),statistics.mean(vals['cfmse']),statistics.stdev(vals['cfmse']),p,ck,statistics.mean(vals['seconds'])*1000,statistics.stdev(vals['seconds'])*1000])
        for s in (20260928,20260929,20260930):
            x=runs[str(s)]; seedrows.append([display,s,x['metrics']['mse'],x['metrics']['cmse'],x['metrics']['cfmse'],x['metrics']['seconds']*1000,p,x['checkpoint_bytes']/1048576])

    # Dashboard
    ws=wb.create_sheet('DASHBOARD'); style(ws,'IDD-PeD / MTN Research Project Tracker','Research-lab project record • completed artifacts only • current proposed POC: Context-Aware MTN with Encoder Self-Attention Sharing')
    cards=[('Project status','12 / 12 unified runs completed',TEAL),('Current proposed POC','Context-Aware MTN with Encoder Self-Attention Sharing',BLUE),('Parameter change','−6.07% versus Context-Aware MTN',TEAL),('MSE change','−5.01% versus Context-Aware MTN',TEAL),('Latency change','+3.93% versus Context-Aware MTN',ORANGE),('Dynamic fusion status','Evaluated separately; not selected',ORANGE)]
    for i,(h,v,c) in enumerate(cards):
        col=1+(i%3)*3; row=4+(i//3)*3; ws.merge_cells(start_row=row,start_column=col,end_row=row,end_column=col+1); ws.cell(row,col,h).font=Font(name='Aptos',bold=True,color='FFFFFF'); ws.cell(row,col).fill=PatternFill('solid',fgColor=c); ws.cell(row,col).alignment=Alignment(horizontal='center'); ws.merge_cells(start_row=row+1,start_column=col,end_row=row+1,end_column=col+1); ws.cell(row+1,col,v).font=Font(name='Aptos Display',size=14,bold=True,color=NAVY); ws.cell(row+1,col).fill=PatternFill('solid',fgColor='F7FAFC'); ws.cell(row+1,col).alignment=Alignment(horizontal='center')
    note(ws,'A11','Methods: Context-Aware MTN; Context-Aware MTN with Encoder Self-Attention Sharing; MTN with Dynamic Multimodal Fusion; Dynamic Multimodal Fusion MTN with Encoder Self-Attention Sharing.',NAVY); ws.merge_cells('A11:H11')
    table(ws,13,['Model','MSE mean','Parameters','Latency ms'],[[x[0],x[1],x[7],x[9]] for x in unified],'DashboardMetrics',[16,16,16,16])
    # seed chart staging columns
    ws['J13']='Seed'; [ws.cell(13,11+i,m) for i,m in enumerate(model_map)]
    for rr,s in enumerate((20260928,20260929,20260930),14):
        ws.cell(rr,10,s)
        for cc,key in enumerate(model_map,11): ws.cell(rr,cc,next(x[2] for x in seedrows if x[0]==key and x[1]==s))
    def chart(title, col, anchor, color):
        c=BarChart(); c.title=title; c.y_axis.title=title.split(' ')[0]; c.x_axis.title='Variant'; c.style=10; c.add_data(Reference(ws,min_col=col,min_row=13,max_row=16),titles_from_data=True); c.set_categories(Reference(ws,min_col=1,min_row=14,max_row=16)); c.height=7; c.width=11; c.series[0].graphicalProperties.solidFill=color; ws.add_chart(c,anchor)
    chart('MSE comparison',2,'A19',BLUE); chart('Parameter count comparison',3,'H19',TEAL); chart('Latency comparison (ms)',4,'A34',ORANGE)
    lc=LineChart(); lc.title='MSE by seed'; lc.style=13; lc.y_axis.title='MSE'; lc.x_axis.title='Seed'; lc.add_data(Reference(ws,min_col=11,max_col=14,min_row=13,max_row=16),titles_from_data=True); lc.set_categories(Reference(ws,min_col=10,min_row=14,max_row=16)); lc.height=7; lc.width=11; ws.add_chart(lc,'H34')
    # Architecture parameter evolution chart includes variants only, not comparable performance
    ws['J19']='Architecture'; ws['K19']='Parameters'; params=[('Original MTN',133700),('Context-Aware MTN',139142),('Context MTN + Encoder Sharing',130694),('MTN + Dynamic Fusion',154007),('Dynamic Fusion + Encoder Sharing',145559)]
    for i,row in enumerate(params,20): ws.cell(i,10,row[0]); ws.cell(i,11,row[1])
    pc=LineChart(); pc.title='Architecture evolution — parameters'; pc.add_data(Reference(ws,min_col=11,min_row=19,max_row=24),titles_from_data=True); pc.set_categories(Reference(ws,min_col=10,min_row=20,max_row=24)); pc.height=7; pc.width=11; ws.add_chart(pc,'A49')
    for col in ('J','K','L','M','N'):
        ws.column_dimensions[col].hidden=True

    # Timeline
    ws=wb.create_sheet('PROJECT TIMELINE'); style(ws,'Project Timeline','Chronological project history; dates were not recorded in the available artifacts.')
    timeline=[['N/A / not recorded','IDD-PeD repository setup','Make the repository runnable locally','Local project workspace','Repository / environment','Completed'],['N/A / not recorded','MTN implementation','Verify the base method','Original MTN pipeline available','MTN code','Completed'],['N/A / not recorded','Context branch','Use native IDD-PeD scene/occlusion labels','Version B created','Context architecture / POC','Completed'],['N/A / not recorded','DynMM study','Investigate dynamic multimodal fusion','Version F created and gradient-validated','F architecture package','Completed'],['N/A / not recorded','Architecture diagrams','Provide editable supervisor-facing diagrams','Draw.io, SVG, PNG and PDF deliverables','Architecture packages','Completed'],['N/A / not recorded','Weight-sharing audit','Reduce parameters without changing task','Full and targeted sharing studies','Audit reports','Completed'],['N/A / not recorded','C4 validation','Validate encoder self-attention sharing','C4 selected for unified validation','C4 reports','Completed'],['N/A / not recorded','Unified experiment','Compare B/C4/F/F+C4 under one protocol','12 matched runs','final_four_way_results.json','Completed'],['N/A / not recorded','Current proposal','Record measured POC outcome','C4 proposed for supervisor review','C4 final architecture','Current POC']]
    table(ws,4,['Date / stage','Task','Why it was done','What changed','Output','Status'],timeline,'TimelineTable',[20,24,30,28,27,16])

    # Requirements
    ws=wb.create_sheet('REQUIREMENTS'); style(ws,'Supervisor Requirements Traceability','Each item is documented from completed artifacts; no new experiment is implied.')
    reqs=[['Run IDD-PeD locally','Repository and local POC pipeline','Local workspace / reports','Completed','CPU POC used Farneback flow'],['Implement MTN','Original MTN verified','MTN source and POC outputs','Completed','Base architecture A'],['Study DynMM','Implemented Version F','F architecture package','Completed','Evaluated separately'],['Add scene/context labels','Six native inputs in B/F','Context diagrams and code','Completed','No day/night or illumination'],['Test with/without context','Early A/B POC','Decision report','Completed','Not directly comparable to unified run'],['Create architecture diagram','Native Draw.io deliverables','C4 and F packages','Completed','Editable source present'],['Document architectural changes','Architecture/module descriptions','Markdown/PDF artifacts','Completed','Paper-ready material'],['Investigate parameter sharing','Full and targeted studies','Weight-sharing reports','Completed','C4 carried forward'],['Try attention-layer sharing','C1/C2a/C2b/C4','Targeted sharing table','Completed','C3/C5 not run'],['Try cross-attention sharing','Encoder cross-attention considered','Targeted audit','N/A / not recorded','No measured C3/C5 result'],['Test encoder sharing','C4 Q/K/V/O sharing','Unified run','Completed','Only encoder self_attn shared'],['Compare architectures','B/C4/F/F+C4','Unified 12-run results','Completed','Same protocol'],['Produce final architecture figure','C4 package','final_architecture folder','Completed','C4 is current POC'],['Prepare paper-ready description','Architecture text and tables','PDF/Markdown packages','Ready','Overleaf integration pending']]
    table(ws,4,['Supervisor requirement','Implementation','Evidence / output','Status','Notes'],reqs,'RequirementsTable',[28,32,28,18,38])

    # Architecture evolution
    ws=wb.create_sheet('ARCHITECTURE EVOLUTION'); style(ws,'Architecture Evolution','Variants are distinct experimental candidates and must not be mixed across different POC protocols.')
    evo=[['Original MTN','Baseline trajectory prediction','[B,15,4] trajectory + flows','No scene context','No added fusion','No sharing',133700,'Early POC MSE 1242.98','Reference only'],['Context-Aware MTN','Add native scene and occlusion context','Trajectory + six context inputs','96-D → 32-D gated token','Append token after encoder','No sharing',139142,'Unified MSE 1137.11 ± 20.16','Context reference'],['Context-Aware MTN with Encoder Self-Attention Sharing','Reduce unique encoder self-attention parameters','Same trajectory and six context inputs','Same gated context branch','Append token after encoder','Encoder self-attention Q/K/V/O only',130694,'Unified MSE 1080.16 ± 93.33','Current proposed POC'],['MTN with Dynamic Multimodal Fusion','Test adaptive trajectory-context fusion','Trajectory + six tokens + global context','Tokens [B,6,32] + global [B,32]','Identity/additive/cross-attention gate','No sharing',154007,'Unified MSE 1154.78 ± 9.95','Experimental candidate'],['Dynamic Multimodal Fusion MTN with Encoder Self-Attention Sharing','Combine dynamic fusion and encoder sharing','Same as dynamic fusion model','Same dynamic fusion context','Dynamic fusion before decoder','Encoder self-attention Q/K/V/O only',145559,'Unified MSE 1126.67 ± 30.84','Experimental candidate']]
    table(ws,4,['Architecture','Purpose','Input','Context representation','Fusion / placement','Weight sharing','Parameters','Measured result','Decision'],evo,'EvolutionTable',[34,28,28,30,30,31,16,25,22])

    # baseline/context
    ws=wb.create_sheet('BASELINE & CONTEXT'); style(ws,'Baseline & Context POCs','Early results below use different budgets and sample counts; do not compare their absolute values with the later unified experiment.')
    ab=[['Original MTN','Trajectory-only baseline','133,700','1242.98','1225.17','4093.85','69.24','1 seed; 12 epochs; 10 held-out trajectories; GP6 video 3; CPU Farneback','Early POC'],['Context-Aware MTN','MTN with six native context inputs','139,142','1113.21','1094.57','3646.51','61.41','1 seed; 12 epochs; 10 held-out trajectories; GP6 video 3; CPU Farneback','Early POC']]
    table(ws,4,['Architecture','Definition','Parameters','MSE','C-MSE','CF-MSE','Latency ms','Protocol','Comparability'],ab,'ABTable',[28,28,16,14,14,14,14,50,20])
    note(ws,'A9','Implemented B context inputs: intersection_type, signalized_type, road_type, location_type, motion_direction, observed_occlusion. Day/night and illumination were discussed but not verified as final implemented inputs.',ORANGE); ws.merge_cells('A9:I10')
    note(ws,'A12','B flow: six 16-D embeddings → concatenation [B,96] → context projection [B,32] → original context admission gate → gated token → append after encoder memory before decoder.',TEAL); ws.merge_cells('A12:I13')

    # weight sharing
    ws=wb.create_sheet('WEIGHT SHARING'); style(ws,'Weight-Sharing Investigation','Parameter reduction ≠ FLOP reduction. Attention executions remain unless layers/branches are removed, which was not done for C4.')
    full=[['B (early reference)',139142,1113.21,'47.20 ms','Early sharing comparison baseline'],['Full encoder + decoder sharing',53894,1339.31,'50.13 ms','−61.3% parameters; MSE worsened; rejected']]
    table(ws,4,['Variant','Parameters','MSE','Latency','Conclusion'],full,'FullSharingTable',[32,18,18,18,50])
    target=[['Context-Aware MTN','No sharing',139142,972.51,949.45,2802.67,47.79,'Reference'],['Decoder source-attention: Q/K/V/O shared','Decoder src_attn Q/K/V/O',130694,1081.44,1049.76,3369.58,53.68,'Worse error and slower'],['Decoder source-attention: Q/K/V shared','Decoder src_attn Q/K/V',132806,1188.92,1169.76,3940.10,56.30,'Worse error and slower'],['Decoder source-attention: K/V shared','Decoder src_attn K/V',134918,1152.49,1133.15,3807.02,47.99,'Worse error'],['Encoder self-attention: Q/K/V/O shared','Encoder self_attn Q/K/V/O',130694,819.99,798.82,2441.16,49.56,'Selected for further validation'],['Decoder self-attention sharing','Not run — current decoder has no active decoder self-attention','N/A / not recorded','N/A / not recorded','N/A / not recorded','N/A / not recorded','N/A / not recorded','Not run'],['Attention-head sharing','Not run — genuine head sharing requires new parameterization','N/A / not recorded','N/A / not recorded','N/A / not recorded','N/A / not recorded','N/A / not recorded','Not run']]
    table(ws,10,['Sharing experiment','Exact sharing scope','Parameters','MSE','C-MSE','CF-MSE','Latency ms','Decision'],target,'TargetSharingTable',[38,46,18,15,15,15,15,32])
    note(ws,'A20','Selected encoder-sharing design: ONLY Encoder.layers[0..2].self_attn Q/K/V/O are shared. Encoder FFNs/norms/src_attn/cross_attn, decoder, context branch and output head remain independent. All three encoder attention executions still run; FLOPs are not reduced.',RED); ws.merge_cells('A20:H22')

    # DynMM
    ws=wb.create_sheet('DYNMM – FUSION'); style(ws,'Dynamic Multimodal Fusion (F)','F is an evaluated separate candidate, not the current selected C4 architecture.')
    frows=[['Why investigated','Test whether trajectory-context fusion could adapt per sample',''],['Trajectory input','Trajectory memory [B,15,32]','MTN encoder output'],['Context tokens','Six context tokens [B,6,32]','Each of six 16-D embeddings projected to 32-D'],['Global context','[B,96] → [B,32]','Separate from six-token cross-attention input'],['Identity candidate','Trajectory unchanged [B,15,32]','Candidate 1'],['Additive candidate','Broadcast global context + trajectory [B,15,32]','Candidate 2 + LayerNorm'],['Cross-attention candidate','Q trajectory; K/V six context tokens','MHA + residual + norm + FFN; candidate 3'],['Dynamic gate','pooled trajectory [B,32] + global context [B,32] → [B,64] → 64→32→3','Hard Gumbel-Softmax train; argmax/one-hot eval'],['Gradient validation','Q/K/V/O all received non-zero gradients','Technically valid six-token design'],['Computation','All three candidates computed before selection','No branch skipping; no resource-aware loss'],['Unified result','MSE 1154.78 ± 9.95; parameters 154,007; latency 50.68 ± 2.48 ms','Did not improve measured metrics over B']]
    table(ws,4,['Topic','Implemented evidence','Notes'],frows,'FusionTable',[26,55,55])

    # unified
    ws=wb.create_sheet('UNIFIED 12-RUN RESULTS'); style(ws,'Unified Final Experiment — Aggregate Results','Fixed 32/8/24 split • CPU Farneback • Adam 0.001 • batch 8 • MSE loss • one epoch • three seeds • all 12 runs completed.')
    urows=[[x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10]] for x in unified]
    table(ws,4,['Architecture','MSE mean','MSE std','C-MSE mean','C-MSE std','CF-MSE mean','CF-MSE std','Parameters','Checkpoint MiB','Latency mean ms','Latency std ms'],urows,'UnifiedTable',[44,16,14,16,14,16,14,16,18,18,16])
    for col in ('B','D','F','H'):
        ws.conditional_formatting.add(f'{col}5:{col}8',ColorScaleRule(start_type='min',start_color='63BE7B',mid_type='percentile',mid_value=50,mid_color='FFEB84',end_type='max',end_color='F8696B'))
    ws.conditional_formatting.add('J5:J8',ColorScaleRule(start_type='min',start_color='63BE7B',mid_type='percentile',mid_value=50,mid_color='FFEB84',end_type='max',end_color='F8696B'))
    note(ws,'A11','Interpretation: The context-aware architecture with encoder self-attention sharing had the lowest mean error metrics and lowest parameter count in this specific unified POC. This is not a claim of universal superiority or an official benchmark.',ORANGE); ws.merge_cells('A11:K12')
    definitions=[['Context-Aware MTN','Six native context inputs are embedded, projected to 32-D, gated, and appended after encoder memory.','No dynamic fusion','No sharing'],['Context-Aware MTN with Encoder Self-Attention Sharing','Same six-input gated context branch as Context-Aware MTN.','No dynamic fusion','Only encoder self-attention Q/K/V/O shared across the three encoder layers'],['MTN with Dynamic Multimodal Fusion','Six context tokens [B,6,32] plus global context [B,32].','Identity, additive, and cross-attention candidates with dynamic selection.','No sharing'],['Dynamic Multimodal Fusion MTN with Encoder Self-Attention Sharing','Same six-token dynamic fusion design.','Same dynamic fusion design.','Only encoder self-attention Q/K/V/O shared across the three encoder layers']]
    table(ws,14,['Architecture definition','Context inputs / representation','Fusion','Weight-sharing scope'],definitions,'UnifiedDefinitions',[44,52,42,50])

    # seed
    ws=wb.create_sheet('PER-SEED RESULTS'); style(ws,'Unified Final Experiment — Per-Seed Results','Exact saved per-run metrics from final_four_way_results.json; latency is measured CPU inference time in ms.')
    table(ws,4,['Model','Seed','MSE','C-MSE','CF-MSE','Latency ms','Parameters','Checkpoint MiB'],seedrows,'PerSeedTable',[14,16,16,16,16,18,16,18])
    ws.conditional_formatting.add('C5:C16',ColorScaleRule(start_type='min',start_color='63BE7B',mid_type='percentile',mid_value=50,mid_color='FFEB84',end_type='max',end_color='F8696B'))

    # comparison
    ws=wb.create_sheet('COMPARISON'); style(ws,'Change Relative to Context-Aware MTN — Unified POC','Negative error/parameter percentage indicates reduction. Positive latency percentage indicates slower CPU inference.')
    changes=[['Context-Aware MTN with Encoder Self-Attention Sharing','−5.01%','−4.84%','−2.82%','−6.07%','+3.93%','Best mean error and fewer parameters; slower'],['MTN with Dynamic Multimodal Fusion','+1.55%','+1.52%','+4.27%','+10.68%','+11.18%','Worse metrics, more parameters and slower'],['Dynamic Multimodal Fusion MTN with Encoder Self-Attention Sharing','−0.92%','−0.81%','+4.72%','+4.61%','+14.23%','Mixed error result; more parameters and slowest']]
    table(ws,4,['Architecture compared with Context-Aware MTN','MSE change','C-MSE change','CF-MSE change','Parameters change','Latency change','Factual reading'],changes,'ComparisonTable',[48,18,18,18,20,18,55])
    note(ws,'A10','No overall score or ranking is used. Parameter reduction ≠ FLOP reduction ≠ latency reduction. Encoder self-attention sharing still executes all encoder attention operations; dynamic fusion computes all three candidates before selection.',RED); ws.merge_cells('A10:G12')

    # final architecture
    ws=wb.create_sheet('FINAL ARCHITECTURE'); style(ws,'Current Proposed POC Architecture — Context-Aware MTN with Encoder Self-Attention Sharing','Proposed only for this completed CPU Farneback POC, pending supervisor review and broader validation.')
    flow=[['1','Observed boxes','[B,15,4]'],['2','Source embedding + positional encoding','[B,15,32]'],['3','MTN Encoder ×3','Trajectory memory [B,15,32]'],['4','Cross-layer encoder self-attention sharing','Only Encoder.layers[0..2].self_attn Q/K/V/O share unique parameters'],['5','Six context inputs','intersection_type, signalized_type, road_type, location_type, motion_direction, observed_occlusion'],['6','Context embeddings','6 × 16-D → concatenate [B,96] → context projection [B,32]'],['7','Context admission gate','Gated context token [B,1,32]'],['8','Append to memory','Augmented decoder memory [B,16,32]'],['9','MTN Decoder ×3','Existing flow paths unchanged'],['10','Output','Output head + final observed-box residual → future trajectory [B,45,4]']]
    table(ws,4,['Order','Module','Exact flow / dimension'],flow,'FinalArchitectureTable',[12,32,85])
    note(ws,'A17','Dynamic multimodal fusion was evaluated separately and is not part of the current proposed architecture. Encoder self-attention sharing reduces unique parameters, not FLOPs; measured latency increased in the unified POC.',ORANGE); ws.merge_cells('A17:H19')

    # files
    ws=wb.create_sheet('FILES & OUTPUTS'); style(ws,'Files & Outputs Inventory','Recorded references only. “Path not recorded” is used where a reliable exact path was not available.')
    inv=[
        ['Unified validation JSON','final_four_way_results.json','Saved 12-run metrics','Available',r'D:\IDD-PeD-Evaluation\outputs\final_validation\final_four_way_results.json'],
        ['Unified live log','final_four_way_validation.log','Runner progress and completion record','Available',r'D:\IDD-PeD-Evaluation\runs\final_four_way_validation.log'],
        ['Per-run checkpoints','seed folders under final_validation','12 model/seed outputs','Available',r'D:\IDD-PeD-Evaluation\outputs\final_validation'],
        ['C4 final package','C4_Final_Architecture.drawio / pdf','Editable and publication architecture','Available',r'D:\IDD-PeD-Evaluation\outputs\final_architecture'],
        ['C4 description','C4_Final_Architecture_Description.md','Module description','Available',r'D:\IDD-PeD-Evaluation\outputs\final_architecture'],
        ['F architecture package','Dynamic_Multimodal_Fusion_MTN.*','Editable diagram, SVG, PNG, PDF, description','Available',r'D:\IDD-PeD-Evaluation\outputs\multimodal_architecture'],
        ['Weight sharing architecture audit','Weight_Sharing_Architecture_Audit_Report.md','Sharing analysis','Path not recorded','Path not recorded'],
        ['Targeted weight-sharing report','Targeted weight-sharing report','C1/C2/C4 comparison','Path not recorded','Path not recorded'],
        ['MTN architecture mapping','MTN architecture mapping PDF','Implementation mapping','Path not recorded','Path not recorded'],
        ['Improved MTN architecture diagram','Improved MTN architecture diagram','Earlier context design figure','Path not recorded','Path not recorded'],
        ['Runner script','final four-way runner','Resumable 12-run execution','Path not recorded','Path not recorded'],
        ['Final analysis/report','Unified experiment analysis','Supervisor-facing interpretation','Path not recorded','Path not recorded']]
    table(ws,4,['Category','File','Purpose','Status','Path / reference'],inv,'FileInventory',[25,32,38,20,75])

    # Paper
    ws=wb.create_sheet('PAPER – OVERLEAF'); style(ws,'Paper / Overleaf Readiness','Documentation readiness only; Overleaf integration has not been recorded as complete.')
    paper=[['Architecture section','C4 Context-Aware MTN specification','Ready','C4 description / final diagram'],['Figure','Editable Draw.io and PDF figure','Ready','C4 final architecture package'],['Module-wise description','C4 and F module descriptions','Ready','Markdown/PDF packages'],['Ablation table','Unified B/C4/F/F+C4 aggregate table','Ready','Unified results workbook sheet'],['Results section','Measured unified POC results','Ready','12-run artifacts'],['Discussion','Accuracy, parameters, latency and computational notes','Ready','No universal-superiority claim'],['Limitations','CPU Farneback POC constraints','Ready','Limitations sheet'],['Overleaf integration','Place selected figures/text in manuscript','Pending','No integration artifact recorded']]
    table(ws,4,['Paper item','Deliverable','Status','Evidence'],paper,'PaperTable',[28,45,18,50])

    # limitations
    ws=wb.create_sheet('LIMITATIONS'); style(ws,'Limitations','These constraints bound all interpretations of the current POC results.')
    lim=[['Flow method','CPU Farneback used instead of authors’ RAFT flow','Not exact published-pipeline reproduction'],['Evaluation scope','One held-out video','Limited scene/video diversity'],['Test set','24 trajectories','Small POC sample'],['Training budget','One epoch','Not convergence/benchmark training'],['Replicates','Three seeds','Useful but still limited stability evidence'],['Benchmark claim','Not an official IDD-PeD RAFT-based benchmark','Do not compare as published result'],['C4 computation','C4 does not reduce FLOPs','All encoder attention executions still run'],['F computation','F computes all fusion branches before selection','No branch-skipping saving'],['External validity','Broader multi-video validation not completed','Needed if supervisor requests it']]
    table(ws,4,['Area','Limitation','Impact'],lim,'LimitationsTable',[28,55,55])

    # next steps
    ws=wb.create_sheet('NEXT STEPS'); style(ws,'Next Steps','Derived only from current project status; these are review/documentation steps, not unapproved new experiments.')
    steps=[['1','Supervisor review of C4 architecture diagram','Pending','Confirm presentation and proposed POC direction'],['2','Supervisor review of DynMM/F diagram','Pending','Document experimental candidate separately'],['3','Integrate selected architecture into paper/Overleaf','Pending','Architecture documentation and figures are ready'],['4','Broader validation if supervisor requests it','Conditional','No scope/budget is assumed here'],['5','Finalize paper tables/figures','Pending','Use measured results and limitations verbatim']]
    table(ws,4,['Order','Next step','Status','Basis'],steps,'NextStepsTable',[12,50,18,65])

    # global polish
    for ws in wb.worksheets:
        ws.sheet_properties.outlineSummaryBelow=True
        ws.sheet_properties.pageSetUpPr.fitToPage=True; ws.page_setup.fitToWidth=1; ws.page_setup.fitToHeight=0; ws.page_setup.orientation='landscape'
    wb.calculation.fullCalcOnLoad=True; wb.calculation.forceFullCalc=True
    wb.save(OUT)
    # reload verifies that Excel XML package opens cleanly.
    load_workbook(OUT, data_only=False).close()
    print(OUT)

if __name__=='__main__': main()
