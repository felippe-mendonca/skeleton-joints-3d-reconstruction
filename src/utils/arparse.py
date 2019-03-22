import json
from argparse import ArgumentParser
from sys import exit


class ArgumentParserFile(ArgumentParser):
    def __init__(self, parse_from_file=True, *args, **kwargs):

        self._parse_from_file = parse_from_file
        kwargs.update({'add_help': not self._parse_from_file})

        super(ArgumentParserFile, self).__init__(*args, **kwargs)

        if self._parse_from_file:
            self.add_argument(
                '--config-file',
                type=str,
                required=False,
                help="""JSON configuration file with parameters. Arguments passed on
                command line will overwrite configuration file.""")
            self.add_argument('--help', '-h', action='store_true')

            self._initial_args, self._unknown = self.parse_known_args()

    def parse_args(self, args=None, namespace=None):

        if not self._parse_from_file:
            return super(ArgumentParserFile, self).parse_args(args, namespace)

        parsed_from_file = []
        config_file = self._initial_args.config_file

        if self._initial_args.help:
            self.print_help()
            exit(-1)
        elif config_file is not None:
            with open(config_file, 'r') as f:
                args_from_file = json.load(f)

            for arg_key in self._option_string_actions.keys():
                if not arg_key.startswith('--'):
                    continue
                no_dash_key = arg_key.split('--')[-1]
                if no_dash_key in args_from_file:
                    arg_values = args_from_file[no_dash_key]
                    type_arg_values = type(arg_values)
                    if type_arg_values == list:
                        arg_values = list(map(str, arg_values))
                    elif type_arg_values == int:
                        arg_values = [str(arg_values)]
                    elif type_arg_values == bool:
                        arg_values = []
                    else:
                        arg_values = [arg_values]
                    parsed_from_file.extend([arg_key] + arg_values)

        args_to_parse = parsed_from_file + self._unknown + (args or [])
        args_to_parse = None if len(args_to_parse) == 0 else args_to_parse

        return super(ArgumentParserFile, self).parse_args(args=args_to_parse, namespace=namespace)