import html
import re
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

import markdown
from pydantic import BaseModel

from . import message_schema as msg

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a')

IMAGE_REGEX = r'(?P<path>[/\.\w-]+\.(png|jpg))'
AUDIO_REGEX = r'(?P<path>[/\.\w-]+\.(wav|mp3))'
# `sandbox:` and `file://` are common prefix of ChatGPT output.
LINK_REGEX = r'`*\!?\[(?P<alt>.+)\]\((sandbox:)?(file://)?{}\)`*'

with open(Path(__file__).resolve().parent / '../css/chat.css', 'r', encoding='utf-8') as f:
    chat_css = f.read()

def fix_newlines(string):
    string = string.replace('\n', '\n\n')
    string = re.sub(r'\n{3,}', '\n\n', string)
    string = string.strip()
    return string

# avatars come from streamlit chatbot.
USER_AVATAR = '''\
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="#0e1117" xmlns="http://www.w3.org/2000/svg" color="inherit" class="eyeqlp51 st-emotion-cache-fblp2m ex0cdmw0"><path fill="none" d="M0 0h24v24H0V0z"></path><path d="M10.25 13a1.25 1.25 0 11-2.5 0 1.25 1.25 0 012.5 0zM15 11.75a1.25 1.25 0 100 2.5 1.25 1.25 0 000-2.5zm7 .25c0 5.52-4.48 10-10 10S2 17.52 2 12 6.48 2 12 2s10 4.48 10 10zM10.66 4.12C12.06 6.44 14.6 8 17.5 8c.46 0 .91-.05 1.34-.12C17.44 5.56 14.9 4 12 4c-.46 0-.91.05-1.34.12zM4.42 9.47a8.046 8.046 0 003.66-4.44 8.046 8.046 0 00-3.66 4.44zM20 12c0-.78-.12-1.53-.33-2.24-.7.15-1.42.24-2.17.24a10 10 0 01-7.76-3.69A10.016 10.016 0 014 11.86c.01.04 0 .09 0 .14 0 4.41 3.59 8 8 8s8-3.59 8-8z"></path></svg>'''
BOT_AVATAR = '''\
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="#0e1117" xmlns="http://www.w3.org/2000/svg" color="inherit" class="eyeqlp51 st-emotion-cache-fblp2m ex0cdmw0"><rect width="24" height="24" fill="none"></rect><path d="M20 9V7c0-1.1-.9-2-2-2h-3c0-1.66-1.34-3-3-3S9 3.34 9 5H6c-1.1 0-2 .9-2 2v2c-1.66 0-3 1.34-3 3s1.34 3 3 3v4c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-4c1.66 0 3-1.34 3-3s-1.34-3-3-3zm-2 10H6V7h12v12zm-9-6c-.83 0-1.5-.67-1.5-1.5S8.17 10 9 10s1.5.67 1.5 1.5S9.83 13 9 13zm7.5-1.5c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5.67-1.5 1.5-1.5 1.5.67 1.5 1.5zM8 15h8v2H8v-2z"></path></svg>
'''

def replace_blockquote(m):
    return m.group().replace('\n', '\n> ').replace('\\begin{blockquote}', '').replace('\\end{blockquote}', '')


def convert_to_markdown(string):
    if string == '<|BEGIN-VISIBLE-CHAT|>':
        return ''

    # Blockquote
    string = re.sub(r'(^|[\n])&gt;', r'\1>', string)
    pattern = re.compile(r'\\begin{blockquote}(.*?)\\end{blockquote}', re.DOTALL)
    string = pattern.sub(replace_blockquote, string)

    # Code
    string = string.replace('\\begin{code}', '```')
    string = string.replace('\\end{code}', '```')
    string = re.sub(r'(.)```', r'\1\n```', string)

    result = ''
    is_code = False
    for line in string.split('\n'):
        if line.lstrip(' ').startswith('```'):
            is_code = not is_code

        result += line
        if is_code or line.startswith('|'):  # Don't add an extra \n for tables or code
            result += '\n'
        else:
            result += '\n\n'

    result = result.strip()
    if is_code:
        result += '\n```'  # Unfinished code block

    # Unfinished list, like "\n1.". A |delete| string is added and then
    # removed to force a <ol> or <ul> to be generated instead of a <p>.
    if re.search(r'(\n\d+\.?|\n\*\s*)$', result):
        delete_str = '|delete|'

        if re.search(r'(\d+\.?)$', result) and not result.endswith('.'):
            result += '.'

        result = re.sub(r'(\n\d+\.?|\n\*\s*)$', r'\g<1> ' + delete_str, result)

        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])
        pos = html_output.rfind(delete_str)
        if pos > -1:
            html_output = html_output[:pos] + html_output[pos + len(delete_str):]
    else:
        html_output = markdown.markdown(result, extensions=['fenced_code', 'tables'])

    # Unescape code blocks
    pattern = re.compile(r'<code[^>]*>(.*?)</code>', re.DOTALL)
    html_output = pattern.sub(lambda x: html.unescape(x.group()), html_output)

    return html_output


def chat_html_wrapper(history):
    history = history['visible']

    output = f'<style>{chat_css}</style><div class="chat" id="chat"><div class="messages">'

    for i, _row in enumerate(history):
        row = [convert_to_markdown(entry) for entry in _row]

        if row[0]:  # don't display empty user messages
            output += f"""
                  <div class="message">
                    <div class="circle-you">
                      {USER_AVATAR}
                    </div>
                    <div class="text">
                      <div class="message-body">
                        {row[0]}
                      </div>
                    </div>
                  </div>
                """

        if row[1]:
            output += f"""
                  <div class="message">
                    <div class="circle-bot">
                      {BOT_AVATAR}
                    </div>
                    <div class="text">
                      <div class="message-body">
                        {row[1]}
                      </div>
                    </div>
                  </div>
                """

    output += '</div></div>'
    return output


def display_image(path):
    path = Path(path).absolute().relative_to(Path.cwd())
    return f'<div><img src="file/{str(path)}" alt="Image"></div>'


def display_audio(path):
    path = Path(path).absolute().relative_to(Path.cwd())
    return f'<div><audio controls="controls"><source src="file/{str(path)}"></audio></div>'


def sub_image_path(match_obj: re.Match):
    return display_image(match_obj.groupdict()['path'])


def sub_audio_path(match_obj: re.Match):
    return display_audio(match_obj.groupdict()['path'])


def tool_to_html(input: msg.ToolInput, output: Optional[msg.ToolOutput] = None):
    tool = input.name

    if input.thought:
        html = f'<div class="thought">{convert_to_markdown(input.thought)}</div>'
    else:
        html = ''

    html += '<details class="tool">'
    if output is None:
        html += f'<summary>Executing <em>{tool}</em> ...</summary>'
    elif output.outputs is not None:
        html += f'<summary><em>{tool}</em></summary>'
    else:
        html += f'<summary class="tool-error">Failed to execute <em>{tool}</em></summary>'

    if input.args:
        args = deepcopy(input.args)
        def replace_arg(data: dict):
            if data['type'] == 'text':
                return repr(data['content'])
            else:
                return '<em>path</em>'

        args = ', '.join(f'{k}={replace_arg(v)}' for k, v in args.items())
        html += f'<div class="tool-args">Args: {args}</div>'

    display = ''
    if output and output.outputs is not None:
        text_output = ', '.join(
            str(out['content']) if out['type'] == 'text' else f'<em>{out["type"]}</em>'
            for out in output.outputs)
        html += f'<div class="tool-response">Response: {text_output}'
        for out in output.outputs:
            if out['type'] == 'image':
                display += display_image(out['content'])
            elif out['type'] == 'audio':
                display += display_audio(out['content'])
            elif out['type'] == 'file':
                display += f'<div>{out["content"]}</div>'
        html += '</div>'
    elif output and output.error is not None:
        html += f'<div class="tool-error">Error: {output.error}</div>'

    html += '</details>'
    html += display
    return html

def reply_to_html(steps: List[BaseModel]) -> str:
    html = ''
    loading = False
    for i, step in enumerate(steps):
        if isinstance(step, msg.ToolInput):
            if len(steps) > i + 1 and isinstance(steps[i + 1], msg.ToolOutput):
                loading = False
                response = steps[i + 1]
            else:
                loading = True
                response = None
            html += tool_to_html(input=step, output=response)
        if isinstance(step, msg.Answer):
            loading = False
            html += '<div class="final-answer">'
            answer = re.sub(LINK_REGEX.format(IMAGE_REGEX), sub_image_path, step.text)
            answer = re.sub(LINK_REGEX.format(AUDIO_REGEX), sub_audio_path, answer)
            html += convert_to_markdown(answer)
            html += '</div>'
        if isinstance(step, msg.Error):
            loading = False
            html += '<div class="error-box">'
            html += f'<div class="error-title">{step.type}</div>'
            if step.reason:
                reason = step.reason
                if len(reason) > 350:
                    reason = reason[:350] + '...'
                html += f'<div class="error-reason">{reason}</div>'
            html += '</div>'
    if loading:
        html += '<p><em>Is thinking...</em></p>'
    return html
