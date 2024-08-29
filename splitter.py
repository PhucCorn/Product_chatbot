import re
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

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
        for doc in docs:
            chunks = []
            text = doc.page_content
            chapter_matches = [i.group() for i in list(re.finditer(self.patterns[0], text))]
            section_matches = [i.group() for i in list(re.finditer(self.patterns[1], text))]
            subsection_matches = [i.group() for i in list(re.finditer(self.patterns[2], text))]
            combined_pattern = "|".join(self.patterns)
            matches = list(re.finditer(combined_pattern, text))
            chapter = ''
            section = ''
            subsection = ''
            subsubsection = ''
            tree = []
            for i in range(len(matches)):
                start_idx = matches[i].start()+len(matches[i].group())
                end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                content = text[start_idx:end_idx].strip()
                if matches[i].group() in chapter_matches:
                    chapter = matches[i].group()
                    section = ''
                    subsection = ''
                    subsubsection = ''
                    tree += [{'summary':content, 'subtree':[]}]
                elif matches[i].group() in section_matches:
                    section = matches[i].group()
                    subsection = ''
                    subsubsection = ''
                    tree[-1][]
                elif matches[i].group() in subsection_matches:
                    subsection = matches[i].group()
                    subsubsection = ''
                else:
                    subsubsection = matches[i].group()
                chunks.append(Document(page_content=chunk, metadata={'chapter':chapter, 'section':section, 'subsection':subsection, 'subsubsection':subsubsection}))
        return chunks