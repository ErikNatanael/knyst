use proc_macro2::Ident;
use quote::{format_ident, quote};
use syn::{
    parse::Parse, parse_macro_input, spanned::Spanned, FnArg, ImplItem, ItemImpl, Meta, Pat,
    PatIdent, Path, Result, ReturnType, Type, TypePath,
};

#[proc_macro_attribute]
pub fn gen(
    args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let _ = args;
    let gen_impl_data = parse_macro_input!(input as GenImplData);
    gen_impl_data
        .into_token_stream()
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
    // gen_parse(args.into(), input.into()).unwrap_or_else(syn::Error::into_compile_error).into()
}

struct GenImplData {
    /// The last segment of the type_path, which should be used for the function shorthand
    type_ident: Ident,
    process_fn_name: Ident,
    /// Used to refer to the same path for the type as in the original
    type_path: Path,
    inputs: Vec<Ident>,
    outputs: Vec<Ident>,
    parameters: Vec<Parameter>,
    org_item_impl: ItemImpl,
}

impl GenImplData {
    fn into_token_stream(self) -> Result<proc_macro2::TokenStream> {
        let GenImplData {
            type_ident,
            inputs,
            outputs,
            parameters,
            type_path,
            org_item_impl,
            process_fn_name,
        } = self;
        let parameters_in_sig = parameters.into_iter().map(|p| p.ident);
        let num_inputs = inputs.len();
        let num_outputs = outputs.len();
        let type_name_string = type_ident.to_string();
        let match_input_names = inputs.iter().enumerate().map(|(i, name)| {
            let name_string = name.to_string();
            quote! { #i => #name_string, }
        });
        let match_output_names = outputs.iter().enumerate().map(|(i, name)| {
            let name_string = name.to_string();
            quote! { #i => #name_string, }
        });
        let extract_inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, ident)| quote! { let #ident = inputs.get_channel(#i); });
        let extract_outputs = outputs
            .iter()
            .map(|i| quote! { let #i = outputs.next().unwrap(); });

        let handle_name = format_ident!("{type_ident}Handle");
        let handle_functions = inputs.iter().map(|i| {
            quote! {
                fn #i(&self, #i: f32) {
                    todo!();
                }
            }
        });
        Ok(quote! {
                #org_item_impl

                impl knyst_core::gen::Gen for #type_path {
                    fn process(&mut self, ctx: knyst_core::gen::GenContext, resources: &mut knyst_core::resources::Resources) -> knyst_core::gen::GenState {
                        let mut inputs = ctx.inputs;
                        #(#extract_inputs)*

                        let mut outputs = ctx.outputs.iter_mut();
                        #(#extract_outputs)*

                        self.#process_fn_name ( #(#parameters_in_sig),* )
                    }

        fn num_inputs(&self) -> usize {
            #num_inputs
        }
        fn num_outputs(&self) -> usize {
            #num_outputs
        }
        fn init(&mut self, _block_size: usize, _sample_rate: Sample) {}
        fn input_desc(&self, input: usize) -> &'static str {
            match input {
                #(#match_input_names)*
                _ => ""
            }
        }
        fn output_desc(&self, output: usize) -> &'static str {
            match output {
                #(#match_output_names)*
                _ => ""
            }
        }
        fn name(&self) -> &'static str {
            #type_name_string
        }
                }

                // Handle
                #[derive(Copy, Clone, Debug)]
                struct #handle_name {
                    node_id: u64,
                }
                impl #handle_name {
                    #(#handle_functions)*
                }
            })
    }
}

impl Parse for GenImplData {
    fn parse(input: syn::parse::ParseStream) -> Result<Self> {
        let mut item_impl: ItemImpl = input.parse()?;
        let ty = *item_impl.self_ty.clone();
        let Type::Path(TypePath {
            path: type_path, ..
        }) = ty
        else {
            return Err(syn::Error::new(
                ty.span(),
                "this type of impl is not supported",
            ));
        };
        let type_ident = {
            type_path
                .segments
                .first()
                .ok_or(syn::Error::new(type_path.span(), "No segment in path"))?
                .clone()
                .ident
        };

        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut parameters = vec![];
        let mut process_fn_name = None;

        let full_item_span = item_impl.span();

        for item in &mut item_impl.items {
            match item {
                ImplItem::Fn(impl_item_fn) => {
                    let mut remove_attributes = vec![];
                    // Does this function have an attribute we recognise?
                    for (attr_i, attr) in impl_item_fn.attrs.iter_mut().enumerate() {
                        if let Meta::Path(p) = &attr.meta {
                            if let Some(path_segment) = p.segments.first() {
                                match path_segment.ident.to_string().as_ref() {
                                    "process" => {
                                        remove_attributes.push(attr_i);
                                        let ReturnType::Type(_, return_type) =
                                            &impl_item_fn.sig.output
                                        else {
                                            return Err(syn::Error::new(
                                                impl_item_fn.sig.output.span(),
                                                "#[process] method needs to return a GenState",
                                            ));
                                        };
                                        let Type::Path(TypePath {
                                            path: Path { segments, .. },
                                            ..
                                        }) = &**return_type
                                        else {
                                            return Err(syn::Error::new(
                                                return_type.span(),
                                                "#[process] method needs to return a GenState",
                                            ));
                                        };
                                        if segments.last().unwrap().ident.to_string() != "GenState"
                                        {
                                            return Err(syn::Error::new(
                                                return_type.span(),
                                                "#[process] method needs to return a GenState",
                                            ));
                                        }
                                        process_fn_name = Some(impl_item_fn.sig.ident.clone());
                                        for arg in &impl_item_fn.sig.inputs {
                                            if let FnArg::Typed(param) = arg {
                                                let Pat::Ident(PatIdent { ident: name, .. }) =
                                                    &*param.pat
                                                else {
                                                    return Err(syn::Error::new(
                                                        param.span(),
                                                        "Unsupported param",
                                                    ));
                                                };
                                                match *param.ty {
                                                    Type::Reference(ref ty) => {
                                                        match *ty.elem {
                                                            Type::Slice(ref slice_type) => {
                                                                match *slice_type.elem {
                                                                    Type::Path(ref p)
                                                                        if p.path
                                                                            .segments
                                                                            .first()
                                                                            .unwrap()
                                                                            .ident
                                                                            .to_string()
                                                                            == "Sample" =>
                                                                    {
                                                                        ()
                                                                    }
                                                                    _ => {
                                                                        return Err(
                                                                            syn::Error::new(
                                                                                p.span(),
                                                                                "Unknown input",
                                                                            ),
                                                                        );
                                                                    }
                                                                }
                                                                // The type is okay to be an input or output
                                                                if ty.mutability.is_some() {
                                                                    outputs.push(name.clone());
                                                                    parameters.push(Parameter {
                                                                        _ty: ParameterTy::Output,
                                                                        ident: name.clone(),
                                                                    });
                                                                } else {
                                                                    inputs.push(name.clone());
                                                                    parameters.push(Parameter {
                                                                        _ty: ParameterTy::Input,
                                                                        ident: name.clone(),
                                                                    });
                                                                }
                                                            }
                                                            _ => (),
                                                        }
                                                    }
                                                    // TODO: Other types
                                                    _ => (),
                                                }
                                            }
                                        }
                                    }
                                    _ => (),
                                }
                            }
                        }
                    }
                    for i in remove_attributes.iter().rev() {
                        impl_item_fn.attrs.remove(*i);
                    }
                }
                _ => (),
            }
        }

        if process_fn_name.is_none() {
            return Err(syn::Error::new(
                full_item_span,
                "No #[process] method in the block",
            ));
        }

        // let ItemImpl::Type(ItemImpl { ident: type_ident, ty, .. }) = impl_item else {
        //     return Err(syn::Error::new(impl_item.span(), "Invalid impl block"));
        // };
        Ok(GenImplData {
            type_ident,
            type_path,
            inputs,
            outputs,
            parameters,
            org_item_impl: item_impl,
            process_fn_name: process_fn_name.unwrap(),
        })
    }
}

// - `&[Sample]` : input
// - `&mut [Sample]` : output
// - `&Resources` : immutable access to Resources
// - `&mut Resources`: mutable access to Resources
// - `BlockSize`
// - `SampleRate`
// - `&mut MessageSender` : Message output. Direct function call message sending to Gens this is connected to, or adding them to a channel for buffering if messages are sent to a Graph output
enum ParameterTy {
    Input,
    Output,
    // SampleRate,
    // ResourcesShared,
    // ResourcesMut,
    // BlockSize,
}

struct Parameter {
    _ty: ParameterTy,
    ident: Ident,
}
