import re
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from util import *

class PatternSplitter:
    def __init__(self, patterns):
        self.patterns = patterns
        
    
    def split_documents(self, docs):
        chain = summary_chain()
        summaries_queue = []
        def pop_stack(rank, stack, summaries_queue):
            while rank <= stack[-1].metadata['rank']:
                pop = stack.pop()
                if pop.metadata['summary'] == '': 
                    content = content_finder(pop)
                    summary_content = "\n".join([content[0], pop.page_content] + content[1:])
                    summaries_queue += [Document(page_content=summary_content, metadata={'source':docs[0].metadata['source']})]
                    pop.metadata['summary'] = len(summaries_queue)-1
                stack[-1].metadata['subtree'] += [pop]
            return stack
        def add_summaries(branches, summaries):
            for branch in branches:
                idx = branch.metadata['summary']
                if type(idx) is int:
                    if idx>=0: branch.metadata['summary'] = summaries[idx]
                    if branch.metadata['subtree'] != []:
                        branch.metadata['subtree'] = add_summaries(branch.metadata['subtree'], summaries)
            return branches
        split_docs = []
        for doc in docs:
            text = doc.page_content
            chapter_matches = [i.group() for i in list(re.finditer(self.patterns[0], text))]
            section_matches = [i.group() for i in list(re.finditer(self.patterns[1], text))]
            subsection_matches = [i.group() for i in list(re.finditer(self.patterns[2], text))]
            subsubsection_matches = [i.group() for i in list(re.finditer(self.patterns[3], text))]
            combined_pattern = "|".join(self.patterns)
            matches = list(re.finditer(combined_pattern, text))
            chapter = ''
            section = ''
            subsection = ''
            subsubsection = ''
            stack = [Document(page_content='' ,metadata={'source':docs[0].metadata['source'], 'summary':-1, 'rank':0, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
            for i in range(len(matches)):
                start_idx = matches[i].start()+len(matches[i].group())
                end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                content = text[start_idx:end_idx].strip()
                if matches[i].group() in chapter_matches:
                    chapter = matches[i].group()
                    section = ''
                    subsection = ''
                    subsubsection = ''
                    rank = 1
                    stack = pop_stack(rank, stack, summaries_queue)
                    stack += [Document(page_content=content ,metadata={'source':docs[0].metadata['source'], 'summary':'', 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                elif matches[i].group() in section_matches:
                    section = matches[i].group()
                    subsection = ''
                    subsubsection = ''
                    rank = 2
                    stack = pop_stack(rank, stack, summaries_queue)
                    # if content == '':
                    if matches[i].group()[:3]+'.1' in [x[:5] for x in subsection_matches]:
                        stack += [Document(page_content=content ,metadata={'source':docs[0].metadata['source'], 'summary':'', 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                    else:
                        summary_content = "\n".join([chapter[10:], section, content])
                        summaries_queue += [Document(page_content=summary_content, metadata={'source':docs[0].metadata['source'], })]
                        full_content = ": ".join([section[4:],content])
                        stack[-1].metadata['subtree'] += [Document(page_content=content, metadata={'source':docs[0].metadata['source'], 'full_content': full_content, 'summary': len(summaries_queue)-1, 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                elif matches[i].group() in subsection_matches:
                    subsection = matches[i].group()
                    subsubsection = ''
                    rank = 3
                    stack = pop_stack(rank, stack, summaries_queue)
                    # if content == '':
                    if matches[i].group()[:5]+'.1' in [x[:7] for x in subsubsection_matches]:
                        stack += [Document(page_content=content ,metadata={'source':docs[0].metadata['source'], 'summary':'', 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                    else:
                        summary_content = "\n".join([chapter[10:], section[4:], subsection, content])
                        summaries_queue += [Document(page_content=summary_content, metadata={'source':docs[0].metadata['source'], })]
                        full_content = ": ".join([subsection[6:],content])
                        stack[-1].metadata['subtree'] += [Document(page_content=content, metadata={'source':docs[0].metadata['source'], 'full_content': full_content, 'summary': len(summaries_queue)-1, 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                else:
                    subsubsection = matches[i].group()
                    rank = 4
                    summary_content = "\n".join([chapter[10:], section[4:], subsection[6:], subsubsection, content])
                    summaries_queue += [Document(page_content=summary_content, metadata={'source':docs[0].metadata['source'], })]
                    full_content = ": ".join([subsubsection[8:],content])
                    stack[-1].metadata['subtree'] += [Document(page_content=content, metadata={'source':docs[0].metadata['source'], 'full_content': full_content, 'summary': len(summaries_queue)-1, 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
            stack = pop_stack(1, stack, summaries_queue)
            split_docs += [stack[0]]
            while True:
                try:
                    summaries = chain.batch(summaries_queue, config={"max_concurrency": 5})
                    break
                except:
                    print("Limit Token: Hệ thống sẽ chạy lại")
            split_docs = add_summaries(split_docs, summaries)
        return split_docs