import re
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from util import *

class PatternSplitter:
    def __init__(self, patterns):
        self.patterns = patterns
        
    # def split_documents(self, docs):
    #     for doc in docs:
    #         chunks = []
    #         text = doc.page_content
    #         chapter_matches = [i.group() for i in list(re.finditer(self.patterns[0], text))]
    #         section_matches = [i.group() for i in list(re.finditer(self.patterns[1], text))]
    #         subsection_matches = [i.group() for i in list(re.finditer(self.patterns[2], text))]
    #         combined_pattern = "|".join(self.patterns)
    #         matches = list(re.finditer(combined_pattern, text))
    #         chapter = ''
    #         section = ''
    #         subsection = ''
    #         subsubsection = ''
    #         for i in range(len(matches)):
    #             if matches[i].group() in chapter_matches:
    #                 chapter = matches[i].group()
    #                 section = ''
    #                 subsection = ''
    #                 subsubsection = ''
    #             elif matches[i].group() in section_matches:
    #                 section = matches[i].group()
    #                 subsection = ''
    #                 subsubsection = ''
    #             elif matches[i].group() in subsection_matches:
    #                 subsection = matches[i].group()
    #                 subsubsection = ''
    #             else:
    #                 subsubsection = matches[i].group()
    #             start_idx = matches[i].start()+len(matches[i].group())
    #             end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    #             chunk = text[start_idx:end_idx].strip()
    #             chunks.append(Document(page_content=chunk, metadata={'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection}))
    #     return chunks
    
    def split_documents(self, docs):
        chain = summary_chain()
        def pop_stack(rank, stack):
            while rank <= stack[-1].metadata['rank']:
                pop = stack.pop()
                if pop.metadata['summary'] == '': 
                    content = content_finder(pop)
                    pop.metadata['summary'] = chain.invoke(Document(page_content="\n".join([content[0], pop.page_content] + content[1:]), metadata={}))
                stack[-1].metadata['subtree'] += [pop]
            return stack
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
            stack = [Document(page_content='' ,metadata={'summary':'', 'rank':0, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
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
                    stack = pop_stack(rank, stack)
                    stack += [Document(page_content=content ,metadata={'summary':'', 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                    # stack += [Document(page_content='' ,metadata={'summary':vn_2_en("\n".join([chapter, content])), 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                elif matches[i].group() in section_matches:
                    section = matches[i].group()
                    subsection = ''
                    subsubsection = ''
                    rank = 2
                    stack = pop_stack(rank, stack)
                    # if content == '':
                    if matches[i].group()[:3]+'.1' in [x[:5] for x in subsection_matches]:
                        stack += [Document(page_content=content ,metadata={'summary':'', 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                    else:
                        summary_content = "\n".join([chapter[10:], section, content])
                        full_content = ": ".join([section[4:],content])
                        stack[-1].metadata['subtree'] += [Document(page_content=content, metadata={'full_content': full_content, 'summary': chain.invoke(Document(page_content=summary_content, metadata={})), 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                elif matches[i].group() in subsection_matches:
                    subsection = matches[i].group()
                    subsubsection = ''
                    rank = 3
                    stack = pop_stack(rank, stack)
                    # if content == '':
                    if matches[i].group()[:5]+'.1' in [x[:7] for x in subsubsection_matches]:
                        stack += [Document(page_content=content ,metadata={'summary':'', 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                    else:
                        summary_content = "\n".join([chapter[10:], section[4:], subsection, content])
                        full_content = ": ".join([subsection[6:],content])
                        stack[-1].metadata['subtree'] += [Document(page_content=content, metadata={'full_content': full_content, 'summary': chain.invoke(Document(page_content=summary_content, metadata={})), 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
                else:
                    subsubsection = matches[i].group()
                    rank = 4
                    summary_content = "\n".join([chapter[10:], section[4:], subsection[6:], subsubsection, content])
                    full_content = ": ".join([subsubsection[8:],content])
                    stack[-1].metadata['subtree'] += [Document(page_content=content, metadata={'full_content': full_content, 'summary': chain.invoke(Document(page_content=summary_content, metadata={})), 'rank':rank, 'subtree':[], 'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection})]
            stack = pop_stack(1, stack)
            split_docs += [stack[0]]       
        return split_docs