#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)


import os
import re
import bioc
from collections import namedtuple,defaultdict
from typing import List, Dict, Optional
import utils

AnnLine = namedtuple("AnnLine", "annid entity start end text line")
AttrLine = namedtuple("AttrLine", "attrid entity annid attr line")
CommentLine = namedtuple("CommentLine", "commentid annid comment line")
NormLine = namedtuple("NormLine", "normid annid cui line")

class Brat2BioC:
    
    def __init__(self):
    
        self.region_pattern = re.compile('\d{1,2},\d+,\d+')
        self.variant_pattern = re.compile('rs\d+')
        self.VALID_TYPES = ['ENHANCER','PROMOTER', 'REGULATORY_REGION',
                            'TFBS','GENE','TISSUE','DISEASE','SPECIES','VARIANT']
        
        self.REGION_TYPES = ['ENHANCER','PROMOTER','TFBS', 'REGULATORY_REGION']
        self.per_file_issues = defaultdict(list)
        self.per_file_comments = defaultdict(list)
        
    def check_comment(self, file_name : str, ann : bioc.BioCAnnotation):
        
        comment = ann.infons.get('comment')
        if comment is not None:
            variant_comment = re.search(self.variant_pattern, ann.text.lower()) is not None
            region_comment = re.search(self.region_pattern, comment) is not None
            if not variant_comment and not region_comment:
                comment_msg = f"{ann.text} - {ann.infons.get('type')} - {comment}"
                self.per_file_comments[file_name].append(comment_msg)
        
    def get_ann_line(self, line : str) -> Optional[AnnLine]:
        
        ann_line = None
        
        annid,entity_soffset_eoffset,text = line.strip().split("\t")
        entity,soffset,eoffset = entity_soffset_eoffset.split()
        
        if entity != 'MOCK':
            ann_line = AnnLine(annid, entity, int(soffset), int(eoffset), text, line)
        
        return ann_line

    def get_attr_line(self, line : str) -> Optional[AttrLine]:
                
        attr_line = None
        
        att_id,elems = line.strip().split("\t")
        
        elems = elems.split()
        
        if len(elems) == 3:
            attrname, refid, attrvalue = elems
        elif len(elems) == 2:
            attrvalue, refid = elems
            attrname = None
        
        attr_line = AttrLine(att_id, attrname, refid, attrvalue, line)
                                
        return attr_line
    
    def get_norm_line(self, line) -> Optional[NormLine]:
        
        orig_line = line
        
        line = line.replace("Reference","")
        elems = line.strip().split()
        
        normid = elems[0]
        annid = elems[1]
        cui = elems[2]
        if 'MONDO:MONDO:' in cui:
            cui = cui.replace('MONDO:MONDO:', 'MONDO:')
                
        norm_line = NormLine(normid, annid, cui, orig_line)
        
        return norm_line
    
    def get_comment_line(self, line) -> Optional[CommentLine]:
        
        comment_id, name_ref_id, comment = line.strip().split("\t")
        
        ref_id = name_ref_id.replace("AnnotatorNotes ", "")
        
        comment_line  = CommentLine(comment_id, ref_id, comment, line)
        
        return comment_line
        
    def get_parsed_lines(self, lines : List[str]) -> List:
        
        parsed_lines = []
        
        for line in lines:
            if line.startswith("T"):
                parsed_line = self.get_ann_line(line)
            elif line.startswith("A"):
                parsed_line = self.get_attr_line(line)
            elif line.startswith("#"):
                parsed_line = self.get_comment_line(line)
            elif line.startswith("N"):
                parsed_line = self.get_norm_line(line)
                
            parsed_lines.append(parsed_line)
        
        parsed_lines = [l for l in parsed_lines if l is not None]
        
        parsed_lines = sorted(parsed_lines, key = lambda x : x.annid)
    
        return parsed_lines
    
    
    def bioc_ann_sanity_check(self, file_name : str, ann : bioc.BioCAnnotation):
        
        ann_type = ann.infons.get('type')
        
        if ann_type not in self.VALID_TYPES:
            self.per_file_issues[file_name].append('Deprecated type')
        
        if ann_type == 'GENE':
            if ann.infons.get('identifier') is None:
                self.per_file_issues[file_name].append(f'GENE w\o normalization : {ann.text}')
        
        if ann_type == 'SPECIES':
            if ann.infons.get('identifier') is None:
                self.per_file_issues[file_name].append(f'SPECIES w\o normalization: {ann.text}')
                
        if ann_type == 'VARIANT':
            self.variant_sanity_check(ann, file_name)

        if ann_type in self.REGION_TYPES:
            self.region_sanity_check(ann, file_name)
            
        self.error_attribute_sanity_check(ann, file_name)
        
    def error_attribute_sanity_check(self, ann : bioc.BioCAnnotation, file_name : str) -> None:
        
        attribute = ann.infons.get('attribute', [])
                
        if 'ERROR' in attribute:
        
            self.per_file_issues[file_name].append(f'Annotation still with ERROR attribute: `{ann.text}`')
        
    def variant_sanity_check(self, ann : bioc.BioCAnnotation, file_name : str) -> None:
        
        if re.search(self.variant_pattern, ann.text.lower()) is not None:
                ann.infons['identifier'] = ann.text.lower()
        else:
            comment = ann.infons.get('comment')
            if comment is not None:
                ann.infons['identifier'] = comment
            else:
                self.per_file_issues[file_name].append(f'VARIANT w\o normalization: `{ann.text}`')
            
    
    def region_sanity_check(self, ann : bioc.BioCAnnotation, file_name : str) -> None:
        
        comment = ann.infons.get('comment')
        attribute = ann.infons.get('attribute')
        
        if attribute is not None:
            if 'SUFFICIENT' in attribute.split(','):
                if comment is not None:
                    match = re.search(self.region_pattern, comment)
                    if match is not None:
                        coordinates = match.group()
                        genome_build_idx = comment.find('hg') 
                        if genome_build_idx > -1:
                            genome_build = comment[genome_build_idx:genome_build_idx+2]
                        else:
                            genome_build = 'unk'
                    
                        ann.infons['identifier'] = coordinates
                        ann.infons['gb'] = genome_build
                    else:
                        self.per_file_issues[file_name].append(f"REGION SUFFICIENT w\o coordinates: `{ann.text}`")
                else:
                    self.per_file_issues[file_name].append(f"REGION SUFFICIENT w\o coordinates: `{ann.text}`")
            
        else:
            self.per_file_issues[file_name].append(f"REGION w\o (IN)SUFFICIENT attribute: `{ann.text}`")   
            
    
    def build_bioc_ann(self, ann_id : int, text : str, infons : Dict, start : int, end : int) -> bioc.BioCAnnotation:
        
        biocann = bioc.BioCAnnotation()
        biocann.id=str(ann_id)
        biocann.text=text
        
        if infons.get('type') not in self.REGION_TYPES:
            if infons.get('identifier') is not None:
                infons['identifier'] = infons['identifier'].split(':')[-1]

                
        biocann.infons=infons
    
        length = end - start
        locations = [bioc.BioCLocation(offset = start,length = length)]
        biocann.locations=locations
            
        return biocann   
    
        
    def get_annid2infons(self, parsed_lines : List) -> Dict:
        
        annid2infons = {}
        
        annids = [l.annid for l in parsed_lines if isinstance(l, AnnLine)]
        
        for parsed_line in parsed_lines:
            if not isinstance(parsed_line, AnnLine):
                annid = parsed_line.annid
                if annid in annids:
                    if annid not in annid2infons:
                        annid2infons[annid] = {}
                    if isinstance(parsed_line, NormLine):                    
                        annid2infons[annid]['identifier'] = parsed_line.cui
                    if isinstance(parsed_line, AttrLine):
                        # annid2infons[annid]['attribute'] = parsed_line.attr
                        if annid2infons[annid].get('attribute') is None:
                            annid2infons[annid]['attribute'] = []
                        annid2infons[annid]['attribute'].append(parsed_line.attr)
                    if isinstance(parsed_line, CommentLine):
                        annid2infons[annid]['comment'] = parsed_line.comment
                        
                        
        for annid, infons in annid2infons.items():
            if 'attribute' in infons:
                infons['attribute'] = ','.join(infons['attribute'])
        
        return annid2infons
                    

    def bioc_anns_from_parsed_lines(self, file_name : str, parsed_lines : List) -> List[bioc.BioCAnnotation]:
        
        annid2infons = self.get_annid2infons(parsed_lines)
        
        bioc_anns = []
        
        bioc_annid = 1
        
        for parsed_line in parsed_lines:
            if isinstance(parsed_line, AnnLine):
                
                infons = annid2infons.get(parsed_line.annid, {}) 
                infons.update({'type' : parsed_line.entity})
                
                bioc_ann = self.build_bioc_ann(ann_id = bioc_annid,
                                               text = parsed_line.text,
                                               infons = infons,
                                               start = parsed_line.start,
                                               end = parsed_line.end)
                
                self.bioc_ann_sanity_check(ann = bioc_ann, file_name = file_name)
                
                self.check_comment(file_name = file_name , ann = bioc_ann)
                
                bioc_anns.append(bioc_ann)
                
                bioc_annid += 1
        
        
        return bioc_anns
    
    
    def parse_file(self, file_path : str) -> List[bioc.BioCAnnotation]:
        
        file_name = os.path.basename(file_path)
        
        with open(file_path) as infile:
            lines = infile.readlines()
            pmid = file_name.replace('.ann','')
            parsed_lines = self.get_parsed_lines(lines)
            bioc_anns = self.bioc_anns_from_parsed_lines(pmid, parsed_lines) 
        
        return bioc_anns
    
    
    
    def load_clean_text(self, path : str) -> str:
        
        clean = ""
                        
        with open(path) as infile:
            lines  = infile.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('REGIONS:'):
                    break
                else:
                    if line != '':
                        clean += line
        return clean
    
    def build_bioc_doc(self, did : str, text : str, anns : List) -> bioc.BioCDocument:
        
        doc = bioc.BioCDocument()
        doc.id = str(did)
                
        offset = 0
        passage = bioc.BioCPassage()
        passage.text = text
        passage.offset = offset
                
        passage.annotations += anns
                
        doc.passages.append(passage)
                            
        return doc
    

    def create_corpus(self, brat_corpus_path : str, out_path : str):
        
        logger.info('Start converting in BioC format brat files in `{brat_corpus_path}`')
        
        docs = []
                
        for tot, file_path in enumerate(utils.iglob(brat_corpus_path, ext = 'ann', ordered = True)):
            
            did = os.path.basename(file_path).replace('.ann','').replace('PMID-','')
            
            text = self.load_clean_text(file_path.replace('.ann','.txt'))
            
            anns = self.parse_file(file_path)
            
            self.sanity_check_offsets(did = did, text = text, anns = anns)
                        
            self.sanity_check_span_consistency(did = did, text = text, anns = anns)
            
            doc = self.build_bioc_doc(text = text, anns = anns, did = did)
            
            docs.append(doc)
        
        utils.write_collection(outpath=out_path, documents=docs)
        
        collection = bioc.load(out_path)
        
        logger.info(f"Completed creating corpus. TOTAL: {tot+1}: - KEPT: {len(docs)}")
        
        logger.info("Performing sanity checks")
        
        # missing_cuis = self.sanity_check_cuis(collection)
        
        # logger.info("Cross-corpus CUI consistency")
        # cui_consistency = self.sanity_check_crosscorpus_cui_consistency(collection)
        
        # logger.info("Cross-corpus span consistency")
        # span_consistency = self.sanity_check_crosscorpus_span_consistency(collection)
        
        # logger.info("Maybe missing genes")
        # gene_missing = self.sanity_check_missing_genes(collection)
        
        # breakpoint()
        
        # insufficient regions but has `bp`:
            # PMID-2835657
            # PMID-23321477
        
        
    def get_per_file_issues(self):
        
        if len(self.per_file_issues) == 0:
            raise RuntimeError("Issue storage is emtpy! Call `parse_corpus` first!")
            
        for file, issues in self.per_file_issues.items():
            print(file)
            for i in issues:
                print(f'\t{i}')

                
    def sanity_check_offsets(self, did : str, text : str, anns : List[bioc.BioCAnnotation]):
        
        for a in anns:
            if a.text != text[a.total_span.offset:a.total_span.end]:
                raise ValueError(f'Annotation: {a} has wrong offsets')
                
    def sanity_check_span_consistency(self, did : str, text : str, anns : List[bioc.BioCAnnotation]):
        
        # VERIFIED = [15371553, 20864515,19708858,21087445, 21464046, 21494683, 
        #             25582196, 26110280, 26382291, 26728555, 26916345, 27881681,
        #             21527503, 24916375, 25030696, 25223790, 25249570, 28578223,
        #             29385519, 29527594]
        
        VERIFIED = []
        
        ann_cui_pairs = set()
        ann_offest_pairs = set()
        
        for a in anns:
            ann_cui = (a.text, a.infons.get('identifier'))
            ann_cui_pairs.add(ann_cui)
            ann_offset = (a.total_span.offset, a.total_span.end)
            ann_offest_pairs.add(ann_offset)
            
        for a in ann_cui_pairs:
            ann_text = a[0]
            pattern = re.escape(ann_text)
            pattern = re.compile(fr'(?<![^\W_]){pattern}(?![^\W_])')
            # pattern = re.compile(re.escape(ann_text))
            match = re.search(pattern, text)
            
            if match is not None:
                start = match.start()
                end = match.end()
                
                if not any(start >= s and end <= e for (s,e) in ann_offest_pairs):
                    # not in already checked
                    if int(did) not in VERIFIED:    
                        logger.error(f"PMID-{did}: Text span `{ann_text}` is entity but not marked at offset {start}-{end}")

                
            else:
                
                raise ValueError(f"PMID-{did}: Text span `{ann_text}` is entity but not found with regexp")
            
    def sanity_check_cuis(self, collection : bioc.BioCCollection):
        
        atext2dids = defaultdict(set)
        
        for d in collection.documents:
            
            for p in d.passages:
                
                for a in p.annotations:
                    
                    if a.infons.get('type') not in self.REGION_TYPES and a.infons.get('identifier') is None:
                        
                        atext2dids[a.text].add(d.id)
                        
        return atext2dids
    
    
    
    def sanity_check_missing_genes(self, collection : bioc.BioCCollection):
        
        pmid_res = set()
        genes = set()
        
        for d in collection.documents:
            for p in d.passages:
                for a in p.annotations:
                    if a.infons.get('type') == 'GENE':
                        
                        genes.add(a.text)
                    elif a.infons.get('type') in self.REGION_TYPES:
                       pmid_res.add((d.id, a.text))
                        
        missing = {}
        
        for pmid, retext in pmid_res:
                        
            for g in genes:
                match = re.search(re.compile(f'\s{re.escape(g)}\s'), retext)         
                if match is not None:
                    
                    if pmid not in missing:
                        missing[pmid] = defaultdict(set)
                        
                    missing[pmid][retext].add(match.group())
        
        return missing
                
    
    def sanity_check_crosscorpus_span_consistency(self, collection : bioc.BioCCollection):
        
        ann2dids = defaultdict(set)
        did2text = {}
        
        for d in collection.documents:
            for p in d.passages:
                did2text[d.id] = p.text
                for a in p.annotations:
                    if a.text.lower() in ['enhancer', 'promoter', 'enhancers', 'promoters'] or a.text.isnumeric():
                        continue
                    
                    ann = (a.text, a.infons.get('type'), a.total_span.offset, a.total_span.end)
                    ann2dids[ann].add(d.id)
        
        did2anns = defaultdict(set)
        for ann, dids in ann2dids.items():
            for did in dids:
                did2anns[did].add(ann)
        
        
        atext2pmids = defaultdict(set)
        
        for ann, dids in ann2dids.items():
            # exclude documents in which annotation is marked
            tmp_did2text = {did : text for did, text in did2text.items() if did not in dids}
        
            anntext, anntype, annstart, annend = ann 
            
            #exclude documents in which annotation is span within annotation of SAME TYPE
            # if any(a[1] == anntype and (a[2] >= annstart or a[3] <= annend) for a in anns):

            did_to_exclude = []
            for did, anns in did2anns.items():
                if any([a[2] >= annstart or a[3] <= annend for a in anns]):
                    did_to_exclude.append(did)
            
            tmp_did2text = {did : text for did, text in tmp_did2text.items() if did not in did_to_exclude}
            
            pattern = re.compile(f'{re.escape(anntext)}')
            for did, text in tmp_did2text.items():
                if re.search(pattern, text) is not None:
                    atext2pmids[anntext].add(did)
        
        return atext2pmids
                    # logger.warning(f'Document `{did}` has annotation `{anntext}` but it is not marked!')
                    
        
    def sanity_check_crosscorpus_cui_consistency(self, collection : bioc.BioCCollection):
        
        
        # ann2cui = defaultdict(set)
        # ann2did = defaultdict(set)
        # ann2did_cui_type = defaultdict(set)
        
        ann2cui = {}
        ann2did = {}
        
        for d in collection.documents:
            for p in d.passages:
                for a in p.annotations:
                    if a.infons.get('type') not in self.REGION_TYPES:
                        anntext = a.text
                        anncui = a.infons.get('identifier')
                        anntype = a.infons.get('type')
                        
                        if anntype not in ann2cui:
                            ann2cui[anntype] = defaultdict(set)
                        
                        if anntype not in ann2did:
                            ann2did[anntype] = defaultdict(set)
                        
                        ann2cui[anntype][anntext].add(anncui)
                        ann2did[anntype][anntext].add((d.id, anncui))
                        # ann2did_cui_type[anntext].add((d.id, anncui, anntype))
                        
        
        multicuis = {}
        
        for anntype, anntext2anncui in ann2cui.items():
            
            if anntype not in multicuis:
                multicuis[anntype] = {}
            # print(f'\t{anntype}')
                        
            for anntext, anncui in anntext2anncui.items():
    
                if len(anncui) > 1:
                    didscuis = ann2did[anntype].get(anntext)
                    cui2dids = defaultdict(set)
                    for did, cui in didscuis:
                        cui2dids[cui].add(did)
                        
                    
                    if anntext not in multicuis[anntype]:
                        multicuis[anntype][anntext] = {}
                    
                    # print(f"\t\tAnnotation `{anntext}` has multiple cuis over corpus! See documents:")
                    for cui, dids in cui2dids.items():
                        multicuis[anntype][anntext][cui] = dids
                        # print(f"\t\t\t {cui} - {dids}")
        
        return multicuis
    
if __name__ == "__main__":
    
    conf = utils.load_conf()
            
    brat2bioc = Brat2BioC()
    
    # brat_dir = conf['corpus']['brat']
    # bioc_file = conf['corpus']['bioc']
    
    brat_dir = '/home/ele/Desktop/projects/annotation/bte-corpus/server/brat-v1.3_Crunchy_Frog/data/bte-corpus-v2/brat'
    bioc_file = 'test'
    
    brat2bioc.create_corpus(brat_corpus_path = brat_dir, 
                            out_path = bioc_file)
    
    # corpus_pubtator_annotations_path = conf['pubtator']['corpus_bioc']
    # if not os.path.exists(corpus_pubtator_annotations_path):
    #     pmids = [d.id for d in bioc.load(bioc_file).documents]
    #     utils.pubtator_api_fetch_annotated_documents(pmids, outpath = corpus_pubtator_annotations_path)



    
        



