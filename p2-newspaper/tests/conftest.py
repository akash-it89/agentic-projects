from p2news.schema import Block, Line


def line(x0, y0, x1, y1, text="some text", score=0.99) -> Line:
    return Line(box=(x0, y0, x1, y1), text=text, score=score)


def block(id_, box, text="text", column=0, headline=False, line_h=20.0) -> Block:
    return Block(id=id_, lines=[Line(box, text, 0.95)], box=box, line_height=line_h, is_headline=headline,
                 column=column, text=text)
