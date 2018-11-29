import re
from collections import defaultdict


def main():
    books = ["data/Rowling, J.K. - HP 1 - Harry Potter and the Sorcerer's Stone.txt",
             "data/Rowling, J.K. - HP 2 - Harry Potter and the Chamber of Secrets.txt",
             "data/Rowling, J.K. - HP 3 - Harry Potter and the Prisoner of Azkaban.txt",
             "data/Rowling, J.K. - HP 4 - Harry Potter and the Goblet of Fire.txt",
             "data/Rowling, J.K. - HP 5 - Harry Potter and the Order of the Phoenix.txt",
             "data/Rowling, J.K. - HP 6 - Harry Potter and the Half-Blood Prince.txt",
             "data/Rowling, J.K. - HP 7 - Harry Potter and the Deathly Hallows.txt"]

    return extract_info(prepare_text(books))


def prepare_text(books):
    pattern = ("(C H A P T E R [A-Z -]+)\n+" +           # Group 1 selects the chapter number
            "([A-Z \n',.-]+)\\b(?![A-Z]+(?=\.)\\b)" + # Group 2 selects the chapter title but excludes edgs of all caps word beginning first sentence of the chapter
            "(?![a-z']|[A-Z.])" +                     # chapter title ends before lowercase letters or a period
            "(.*?)" +                                 # Group 3 selects the chapter contents
            "(?=C H A P T E R|This book)")            # chapter contents ends with a new chapter or the end of book
    hp = defaultdict(dict)
    for book in books:
        title = book[28:-4]
        with open(book, 'r') as f:
            text = f.read().replace('&rsquo;',"'")
        chapters = re.findall(pattern, text, re.DOTALL)
        chap = 0
        for chapter in chapters:
            chap += 1
            chap_title = chapter[1].replace('\n', '')
            chap_text = (chapter[2][3:].replace('&ldquo;', '"')
                                    .replace('&rdquo;', '"')
                                    .replace('&mdash;', '—'))
            chap_text = re.sub('\n*&bull; [0-9]+ &bull; \n*' + chap_title + ' \n*', '', chap_text, flags=re.IGNORECASE)
            chap_text = re.sub('\n*&bull; [0-9]+ &bull; \s*CHAPTER [A-Z]+ \s*', '', chap_text)
            chap_text = re.sub(' \n&bull; [0-9]+ &bull; \n*', '', chap_text)
            chap_text = re.sub('\n+', '\n', chap_text)
            hp[title]['Chapter ' + str(chap)] = (chap_title, chap_text)
    hp = dict(hp)
    return hp


def extract_info(hp_dict):
    titles = []
    texts = []
    for book in hp_dict:
        for chapter in hp_dict[book]:
            titles.append(hp_dict[book][chapter][0])
            texts.append(hp_dict[book][chapter][1])
    return titles, texts


if __name__ == "__main__":
    main()
