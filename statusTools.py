from os import system


def loading(is_loading, loading_text="Working on it..."):
    print(loading_text if is_loading else 'Done!')


def status(text='new stage...', clear=False):
    system('cls') if clear else None
    print(text)
