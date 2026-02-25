" Setting the python environment to be the global one, need to install npyvim
let g:python3_host_prog = '/opt/conda/bin/python'

" Github copilot
" Check if GitHub Copilot plugin is installed
if empty(glob('~/.config/nvim/pack/github/start/copilot.vim'))
  echo "Installing GitHub Copilot plugin..."
  silent !git clone https://github.com/github/copilot.vim.git ~/.config/nvim/pack/github/start/copilot.vim
  autocmd VimEnter * echom "GitHub Copilot installed. Please restart Neovim."
endif
" At first installation you have to run :Copilot setup. Make sure you have the
" newest version of nvim do that by applying the following commands:
" 1. sudo add-apt-repository ppa:neovim-ppa/stable
" 2. sudo apt update
" 3. sudo apt install neovim


" Define data directory
let data_dir = has('nvim') ? stdpath('data') . '/site' : '~/.vim'
if empty(glob(data_dir . '/autoload/plug.vim'))
  silent execute '!curl -fLo '.data_dir.'/autoload/plug.vim --create-dirs  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
  autocmd VimEnter * PlugInstall --sync | source $MYVIMRC
endif

" Options
set autoindent
set mouse=a

call plug#begin('~/.vim/plugged')

" Install fuzzy finder which allows you to search for any file from the editor
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'
map ; :Files<cr>

" Ale is runs asyncronous linter of choice on the editted file
Plug 'dense-analysis/ale'
let b:ale_linters = {'python': ['flake8']}

Plug 'scrooloose/nerdtree'
"map <C-b> :NERDTreeToggle<cr>

Plug 'jistr/vim-nerdtree-tabs'
map <C-b> :NERDTreeTabsToggle<cr>

Plug 'morhetz/gruvbox'
Plug 'nathanaelkane/vim-indent-guides'
Plug 'itchyny/lightline.vim'

Plug 'dccsillag/magma-nvim', { 'do': ':UpdateRemotePlugins' }

" Add PaperColor theme for light mode
Plug 'NLKNguyen/papercolor-theme'

Plug 'DaikyXendo/nvim-material-icon'

set laststatus=2
if !has('gui_running')
  set t_Co=256
endif

call plug#end()

" General settings
set encoding=UTF-8
set number                                " Show line numbers
set laststatus=2                          " Always show status line
set t_Co=256                              " Enable 256 colors
set termguicolors                         " Enable true color support

" Colorscheme settings for Gruvbox
set background=dark                       " Use Gruvbox dark mode
let g:gruvbox_contrast_dark = 'medium'    " Set medium contrast for dark mode
let g:gruvbox_invert_selection = '0'      " Disable inverted selection
let g:gruvbox_bold = 1                    " Enable bold text
let g:gruvbox_italic = 1                  " Enable italic text
colorscheme gruvbox                       " Apply Gruvbox color scheme

" Indentation settings
set tabstop=4                             " Number of spaces per tab
set shiftwidth=4                          " Number of spaces to use for (auto)indent
set expandtab                             " Use spaces instead of tabs
set autoindent                            " Auto-indent new lines
set smartindent                           " Smart auto-indenting on new lines
set smarttab                              " Makes tabbing smarter

" Lightline configuration (optional)
let g:lightline = { 'colorscheme': 'gruvbox' }
let g:lightline.separator = { 'left': '', 'right': '' }
let g:lightline.subseparator = { 'left': '', 'right': '' }

" Other useful settings
syntax enable                             " Enable syntax highlighting
filetype plugin indent on                 " Enable filetype-specific plugins and indentation
set mouse=a                               " Enable mouse in all modes
set cursorline                            " Highlight the current line

" Disable automatic indentation guide colors for better control
let g:indent_guides_enable_on_vim_startup = 1
let g:indent_guides_color_change_percent = 20


" Disable automatic indentation guide colors for better control
"let g:indent_guides_auto_colors = 0

" Set indentation colors for Gruvbox (dark theme) and PaperColor (light theme)
"autocmd VimEnter,Colorscheme gruvbox :hi IndentGuidesOdd  guibg=#3a3a3a ctermbg=236
"autocmd VimEnter,Colorscheme gruvbox :hi IndentGuidesEven guibg=#4e4e4e ctermbg=237

" Ensure colors are updated when toggling between themes
function! ToggleTheme()
  if g:colors_name ==# 'gruvbox'
    set background=light
    colorscheme PaperColor
    let g:lightline.colorscheme = 'PaperColor'
    " Set indentation colors for PaperColor
"    hi IndentGuidesOdd  guibg=#e8e8e8 ctermbg=254
"    hi IndentGuidesEven guibg=#f2f2f2 ctermbg=253
  else
    set background=dark
    colorscheme gruvbox
    let g:lightline.colorscheme = 'gruvbox'
    " Set indentation colors for Gruvbox
"    hi IndentGuidesOdd  guibg=#3a3a3a ctermbg=236
"    hi IndentGuidesEven guibg=#4e4e4e ctermbg=237
  endif
  call lightline#update()
endfunction

" Map <F8> key to toggle between dark and light themes
nnoremap <F8> :call ToggleTheme()<CR>

" Vimpyter settings
command! TermBelow belowright split | resize 10 | term ++curwin

" Map a key to easily open the terminal in the lower split, for example <F4>
nnoremap <F4> :TermBelow<CR>

" write a refresh function that will be activate whenever <F5> is pressed
function! Refresh()
    :NERDTreeRefreshRoot
endfunction

" Map <F5> key to toggle between dark and light themes
nnoremap <F5> :call Refresh()<CR>

" ctags settings
" set a keystroke that will open the definition of the current word where the
" cursur is pointing in a new tab

" Make CTRL+t open a new tab

" Function to wait for number input and jump to the corresponding tab
function! TabJump()
  let num = input("Jump to tab: ")
  if num =~ '^\d\+$' && num > 0 && num <= tabpagenr('$')
    execute 'tabnext' num
  else
    echo "Invalid tab number"
  endif
endfunction


" Set relative line numbers
set relativenumber

nnoremap <silent><expr> <LocalLeader>r  :MagmaEvaluateOperator<CR>
nnoremap <silent>       <LocalLeader>rr :MagmaEvaluateLine<CR>
xnoremap <silent>       <LocalLeader>r  :<C-u>MagmaEvaluateVisual<CR>
nnoremap <silent>       <LocalLeader>rc :MagmaReevaluateCell<CR>
nnoremap <silent>       <LocalLeader>rd :MagmaDelete<CR>
nnoremap <silent>       <LocalLeader>ro :MagmaShowOutput<CR>

let g:magma_automatically_open_output = v:true
let g:magma_image_provider = "ueberzug"

function! MagmaInitPython()
    MagmaInit python3
    MagmaEvaluateArgument a=5
endfunction

function! MagmaInitCSharp()
    MagmaInit .net-csharp
    MagmaEvaluateArgument Microsoft.DotNet.Interactive.Formatting.Formatter.SetPreferredMimeTypesFor(typeof(System.Object),"text/plain")
endfunction

function! MagmaInitFSharp()
    MagmaInit .net-fsharp
    MagmaEvaluateArgument Microsoft.DotNet.Interactive.Formatting.Formatter.SetPreferredMimeTypesFor(typeof<System.Object>,"text/plain")
endfunction

" Define the commands for Magma initialization
command! MagmaInitPython call MagmaInitPython()
command! MagmaInitCSharp call MagmaInitCSharp()
command! MagmaInitFSharp call MagmaInitFSharp()

