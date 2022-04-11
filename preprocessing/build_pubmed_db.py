
import argparse
import gzip
import multiprocessing
import os
import re
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List

from Bio import Entrez
from dataclasses import dataclass
from tqdm import tqdm
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Build sqlite database from pubmed baseline .xml files.')
    parser.add_argument('--workers', help='Number of workers to use.',
                        default=4, type=int)
    parser.add_argument('--overwrite', help='Recreate the database.',
                        action='store_true')
    return parser.parse_args()

@contextmanager
def open_file(path, mode='rt'):
    """
    If given file ends with .gz use gzip.open, otherwise regular open.
    :param mode: passed to open
    :param path: to file
    """
    is_gzip = path.name.endswith('.gz') if isinstance(path, Path) else path.endswith('.gz')
    if len(mode) == 1:
        mode += 't'
    with gzip.open(path, mode=mode) if is_gzip else open(path, mode=mode) as handle:
        yield handle


def get_files_matching(pattern: str, folder: str) -> List[Path]:
    pattern = re.compile(pattern)
    matching_files = filter(pattern.fullmatch, os.listdir(folder))
    folder = Path(folder)
    return [folder / file for file in matching_files]


@dataclass
class PubMedArticle:
    pmid: int
    title: str
    abstract: Optional[List[str]]


@dataclass
class DeleteCitation:
    pmids: List[int]


class PubMedArticleParser:
    def __init__(self):
        # matches html tags in abstract text and title
        self._html_tags_pattern = re.compile(r'</?(b|i|sup|sub|u)>')

    def _remove_html_tags(self, text):
        return self._html_tags_pattern.sub('', text)

    def _parse_abstract(self, article):
        if 'Abstract' in article:
            abstract_text = article['Abstract']['AbstractText']
            return [self._remove_html_tags(text) for text in abstract_text] or None
        return None

    def _parse_pubmed_article(self, article):
        citation = article['MedlineCitation']
        article = citation['Article']

        pm_id = int(citation['PMID'])
        title = self._remove_html_tags(article['ArticleTitle'])
        abstract_text = self._parse_abstract(article)

        return PubMedArticle(pm_id, title, abstract=abstract_text)

    def _parse_pubmed_book_article(self, article):
        document = article['BookDocument']

        pmid = int(document['PMID'])
        title = self._remove_html_tags(
            document.get('ArticleTitle', document['Book']['BookTitle'])
        )
        abstract_text = self._parse_abstract(document)

        return PubMedArticle(pmid, title, abstract=abstract_text)

    def parse(self, pubmed_xml, delete_citations=False):
        """
        Generator returning parsed PubMedArticle objects from a given .xml
        :param pubmed_xml: Pubmed article set .xml
        :param delete_citations: If True, parse and yield DeleteCitation entities
        """
        print(f'Parsing "{pubmed_xml}"...', file=sys.stderr)
        with open_file(pubmed_xml, 'rb') as handle:
            article_set = Entrez.read(handle, validate=False)

            yield from map(self._parse_pubmed_article, article_set['PubmedArticle'])
            yield from map(self._parse_pubmed_book_article, article_set['PubmedBookArticle'])

            if delete_citations and 'DeleteCitation' in article_set:
                pmids = [int(pmid) for pmid in article_set['DeleteCitation']]
                yield DeleteCitation(pmids)


class PubMedXMLtoRows:
    def __init__(self):
        self.parser = PubMedArticleParser()

    def parse(self, file):
        rows = []
        pmids_to_delete = set()
        for article in self.parser.parse(file, delete_citations=True):
            if isinstance(article, DeleteCitation):
                pmids_to_delete.update(article.pmids)
            else:
                assert isinstance(article, PubMedArticle)
                abstract = None
                if article.abstract is not None:
                    abstract = '\n'.join(article.abstract)
                    # abstract should contain text or None
                    if not abstract:
                        abstract = None
                rows.append((article.pmid, article.title, abstract))

        pmids_to_delete = [[pmid] for pmid in pmids_to_delete]
        return rows, pmids_to_delete


def main(output : str, overwrite : bool, pubmed_dir : str, workers : int  = 2):
    def abort():
        print('Aborted.', file=sys.stderr)
        exit(1)

    def confirm(message):
        try:
            return input(message).lower().strip() in ('y', 'yes')
        except KeyboardInterrupt:
            return False

    if os.path.exists(output):
        if overwrite:
            if not confirm(f'"{output}" already exists. Overwrite? [y/N]'):
                abort()
            os.remove(output)
        elif not confirm(f'"{output}" already exists. Append data? [y/N]'):
            abort()

    if os.path.isdir(pubmed_dir):
        pubmed_files = get_files_matching(r'\S+.xml(.gz)?', pubmed_dir)
        pubmed_files = sorted(pubmed_files, key=lambda file: file.name)
        if not pubmed_files:
            raise ValueError(f'No pubmed files found in "{pubmed_dir}"')
    elif os.path.isfile(pubmed_dir):
        pubmed_files = [Path(pubmed_dir)]
    else:
        raise ValueError(f'Given path {pubmed_dir} is not a file nor a directory!')

    connection = sqlite3.connect(output)
    try:
        # Create table
        connection.execute("""
        CREATE TABLE IF NOT EXISTS pubmed21(
        pmid INTEGER NOT NULL PRIMARY KEY, 
        title TEXT NOT NULL, 
        abstract TEXT
        );
        """)

        parser = PubMedXMLtoRows()
        with multiprocessing.Pool(workers) as pool:
            results = pool.imap(parser.parse, pubmed_files)
            for rows, deletes in tqdm(results, total=len(pubmed_files), dynamic_ncols=True,
                                      desc=f'Writing files to {output}', unit=' files'):
                if rows:
                    connection.executemany("REPLACE INTO pubmed21 VALUES (?, ?, ?)", rows)
                if deletes:
                    connection.executemany("DELETE FROM pubmed21 WHERE pmid = ?", deletes)
                # Save the changes
                connection.commit()
    finally:
        connection.close()

    print('Done.', file=sys.stderr)


if __name__ == '__main__':
    try:
        conf = utils.load_conf()
        args = parse_args()
        
        pubmed_db = os.path.join(conf['pubmed'], 'pubmed21.db')
        pubmed_dir = os.ptah.join(conf['pubmed'], 'raw')
        
        main(output = pubmed_db,
             pubmed_dir = pubmed_dir,
             workers = args.workers,
             overwrite = args.overwrite)
        
    except KeyboardInterrupt:
        print('Execution aborted by user.', file=sys.stderr)
