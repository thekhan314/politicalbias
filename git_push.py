def git_push(repo_name, message):

    message = 'Second colab commit'
    repo_name = 'politicalbias'
    GIT_REPO = 'https://github.com/thekhan314/politicalbias.git'


    from os.path import join

    BASE = 'My Drive/Colab Notebooks/'
    ROOT = '/content/drive/'
    PROJECT_PATH = join(ROOT,BASE)
    REPO_PATH = join(PROJECT_PATH,repo_name)

    from google.colab import drive

    drive.mount(ROOT)

    !git config --global user.email "umarkhan314@gmail.com"
    !git config --global user.name "ColabKhan"

    remote_url = 'https://thekhan314:Bountyhunter22!@github.com/thekhan314/' + repo_name + '.git'
    origin_url = 'https://github.com/thekhan314/politicalbias'
    %cd '{REPO_PATH}'



    !git add . 
    !git commit -m '{message}'
    #!git remote rm origin

    #!git remote add origin '{remote_url}'
    !git push -u origin 